//! Order book with price-time priority matching.
use std::collections::BTreeMap;

/// Price level → queue of orders at that level
#[derive(Debug, Default)]
pub struct OrderBook {
    bids: BTreeMap<u64, Vec<Order>>, // price → orders (highest first)
    asks: BTreeMap<u64, Vec<Order>>, // price → orders (lowest first)
}

#[derive(Debug, Clone)]
pub struct Order {
    pub id: u64,
    pub price: u64,    // in cents (1 = $0.01)
    pub quantity: u64,
    pub side: Side,
    pub timestamp: u64, // millis since epoch
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Side {
    Buy,
    Sell,
}

impl OrderBook {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn resting_count(&self) -> usize {
        self.bids.values().map(|v| v.len()).sum::<usize>()
            + self.asks.values().map(|v| v.len()).sum::<usize>()
    }

    /// Add a limit order to the book
    pub fn add_order(&mut self, order: Order) {
        let side_book = match order.side {
            Side::Buy => &mut self.bids,
            Side::Sell => &mut self.asks,
        };
        side_book
            .entry(order.price)
            .or_default()
            .push(order);
    }

    /// Match incoming market order against resting orders.
    /// Returns list of fills and remaining quantity.
    pub fn match_market(&mut self, side: Side, quantity: u64) -> (Vec<Fill>, u64) {
        let mut fills = vec![];
        let mut remaining = quantity;

        let opposite_book = match side {
            Side::Buy => &mut self.asks,
            Side::Sell => &mut self.bids,
        };

        while remaining > 0 && !opposite_book.is_empty() {
            // Get best price level
            let best_price = if side == Side::Buy {
                *opposite_book.first_entry().unwrap().key()
            } else {
                *opposite_book.last_entry().unwrap().key()
            };

            if let Some(orders) = opposite_book.get_mut(&best_price) {
                let mut filled_all = true;
                for order in orders.iter_mut() {
                    if remaining == 0 {
                        filled_all = false;
                        break;
                    }
                    let fill_qty = remaining.min(order.quantity);
                    fills.push(Fill {
                        price: best_price,
                        quantity: fill_qty,
                    });
                    order.quantity -= fill_qty;
                    remaining -= fill_qty;
                }
                // Remove fully filled orders
                orders.retain(|o| o.quantity > 0);
                if orders.is_empty() {
                    opposite_book.remove(&best_price);
                }
                if !filled_all {
                    break;
                }
            }
        }

        (fills, remaining)
    }
}

#[derive(Debug)]
pub struct Fill {
    pub price: u64,
    pub quantity: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_matching() {
        let mut book = OrderBook::new();

        // Add a sell order at $100
        book.add_order(Order {
            id: 1, price: 10000, quantity: 10,
            side: Side::Sell, timestamp: 1,
        });

        // Buy 5 shares
        let (fills, remaining) = book.match_market(Side::Buy, 5);
        assert_eq!(fills.len(), 1);
        assert_eq!(fills[0].price, 10000);
        assert_eq!(fills[0].quantity, 5);
        assert_eq!(remaining, 0);
    }

    #[test]
    fn test_price_time_priority() {
        let mut book = OrderBook::new();

        // Two sell orders: $100 then $101
        book.add_order(Order { id: 1, price: 10100, quantity: 10, side: Side::Sell, timestamp: 1 });
        book.add_order(Order { id: 2, price: 10000, quantity: 10, side: Side::Sell, timestamp: 2 });

        // Buy should match $100 first (better price)
        let (fills, _) = book.match_market(Side::Buy, 15);
        assert_eq!(fills.len(), 2);
        assert_eq!(fills[0].price, 10000); // better price first
        assert_eq!(fills[1].price, 10100);
    }
}
