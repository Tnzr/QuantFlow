import React, { useEffect, useRef } from "react";
import { View, Animated, StyleSheet } from "react-native";
import THEME from "../../theme/colors";

export default function LoadingSpinner({ size = 40, color = THEME.accent }) {
  const pulse = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    const anim = Animated.loop(
      Animated.sequence([
        Animated.timing(pulse, { toValue: 1, duration: 800, useNativeDriver: true }),
        Animated.timing(pulse, { toValue: 0, duration: 800, useNativeDriver: true }),
      ])
    );
    anim.start();
    return () => anim.stop();
  }, [pulse]);

  const opacity = pulse.interpolate({ inputRange: [0, 1], outputRange: [0.3, 1] });
  const scale = pulse.interpolate({ inputRange: [0, 1], outputRange: [0.8, 1.1] });

  return (
    <View style={[styles.wrap, { width: size, height: size }]}>
      <Animated.View
        style={[
          styles.circle,
          {
            width: size * 0.5,
            height: size * 0.5,
            backgroundColor: color,
            opacity,
            transform: [{ scale }],
          },
        ]}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: { justifyContent: "center", alignItems: "center" },
  circle: { borderRadius: 999 },
});
