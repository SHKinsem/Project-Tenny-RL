#!/usr/bin/env bash
set -euo pipefail

FW="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FQBN="${FQBN:-esp32:esp32:esp32}"
LIB_ROOT="${ARDUINO_LIB_ROOT:-$FW/../.arduino/user/libraries}"

if ! command -v arduino-cli >/dev/null 2>&1; then
  echo "错误: 找不到 arduino-cli。" >&2
  exit 1
fi

for lib in ArduinoJson SCServo; do
  if [[ ! -d "$LIB_ROOT/$lib" ]]; then
    echo "错误: 找不到 $LIB_ROOT/$lib；可用 ARDUINO_LIB_ROOT 指定已有库目录。" >&2
    exit 1
  fi
done

c_string() {
  local value=${1//\\/\\\\}
  value=${value//\"/\\\"}
  printf '%s' "$value"
}

WORK_DIR="$(mktemp -d "$FW/.arduino-compile.XXXXXX")"
trap 'rm -rf "$WORK_DIR"' EXIT
SKETCH_DIR="$WORK_DIR/wl_pro_robot"
mkdir -p "$SKETCH_DIR"
cp "$FW/wl_pro_robot.ino" "$FW/build_opt.h" "$FW/sketch.yaml" "$SKETCH_DIR/"
cp -R "$FW/src" "$FW/data" "$SKETCH_DIR/"

if [[ -n "${WIFI_STA_SSID:-}${WIFI_STA_PASSWORD:-}${WIFI_AP_SSID:-}${WIFI_AP_PASSWORD:-}" ]]; then
  {
    printf '#pragma once\n'
    printf '#define WIFI_STA_SSID "%s"\n' "$(c_string "${WIFI_STA_SSID:-YOUR_WIFI_SSID}")"
    printf '#define WIFI_STA_PASSWORD "%s"\n' "$(c_string "${WIFI_STA_PASSWORD:-YOUR_WIFI_PASSWORD}")"
    printf '#define WIFI_AP_SSID "%s"\n' "$(c_string "${WIFI_AP_SSID:-ESP32-Access-Point}")"
    printf '#define WIFI_AP_PASSWORD "%s"\n' "$(c_string "${WIFI_AP_PASSWORD:-CHANGE_ME_123}")"
  } > "$SKETCH_DIR/src/web/wifi_config.h"
fi

arduino-cli compile --clean \
  --fqbn "$FQBN" \
  --library "$LIB_ROOT/ArduinoJson" \
  --library "$LIB_ROOT/SCServo" \
  --build-property "build.extra_flags=-I$SKETCH_DIR/src -DESP32" \
  "$SKETCH_DIR" "$@"
