/**
 * Date formatting utilities for chart x-axis labels.
 * Supports multiple formats and timezone display.
 */

const TZ = (typeof process !== "undefined" && process.env?.EXPO_PUBLIC_TIMEZONE) || "America/New_York";

function getTzShort() {
  try {
    const parts = new Intl.DateTimeFormat("en-US", {
      timeZone: TZ,
      timeZoneName: "short",
    }).formatToParts(new Date());
    return parts.find((p) => p.type === "timeZoneName")?.value || "EST";
  } catch {
    return "EST";
  }
}

const TZ_SHORT = getTzShort();

/**
 * Format a date string for chart x-axis labels.
 * @param {string} d - Date string (ISO, YYYY-MM-DD, or other parseable format)
 * @param {string} mode - "short" (MM/DD), "time" (HH:MM), "datetime" (MM/DD HH:MM)
 * @returns {string}
 */
export function formatChartDate(d, mode = "short") {
  if (!d) return "";
  const s = String(d);
  if (mode === "short") {
    // YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS -> MM/DD
    if (s.length >= 10) return `${s.slice(5, 7)}/${s.slice(8, 10)}`;
    return s;
  }
  if (mode === "datetime") {
    // Try to parse and format with timezone
    const dt = new Date(s);
    if (!isNaN(dt.getTime())) {
      const mm = String(dt.getMonth() + 1).padStart(2, "0");
      const dd = String(dt.getDate()).padStart(2, "0");
      const hh = String(dt.getHours()).padStart(2, "0");
      const mi = String(dt.getMinutes()).padStart(2, "0");
      return `${mm}/${dd} ${hh}:${mi}`;
    }
    return s.slice(5, 16) || s;
  }
  if (mode === "time") {
    const dt = new Date(s);
    if (!isNaN(dt.getTime())) {
      const hh = String(dt.getHours()).padStart(2, "0");
      const mi = String(dt.getMinutes()).padStart(2, "0");
      return `${hh}:${mi}`;
    }
    return s;
  }
  return s;
}

/**
 * Get the current timezone abbreviation (e.g., "EST", "EDT", "PST").
 */
export function getTimezoneAbbr() {
  return TZ_SHORT;
}

/**
 * Format a full date with timezone for tooltips.
 */
export function formatTooltipDate(d) {
  if (!d) return "";
  const dt = new Date(d);
  if (isNaN(dt.getTime())) return String(d);
  const mm = String(dt.getMonth() + 1).padStart(2, "0");
  const dd = String(dt.getDate()).padStart(2, "0");
  const yyyy = dt.getFullYear();
  const hh = String(dt.getHours()).padStart(2, "0");
  const mi = String(dt.getMinutes()).padStart(2, "0");
  return `${mm}/${dd}/${yyyy} ${hh}:${mi} ${TZ_SHORT}`;
}
