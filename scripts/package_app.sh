#!/bin/sh
set -eu

if [ "$(uname -s)" != "Darwin" ]; then
    echo "Сборка AirLLM.app доступна только на macOS." >&2
    exit 1
fi

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
APP_DIR="$ROOT/dist/AirLLM.app"
CONTENTS="$APP_DIR/Contents"

cd "$ROOT"
rm -rf "$APP_DIR"
mkdir -p "$CONTENTS/MacOS" "$CONTENTS/Resources"

CGO_ENABLED=1 go build -trimpath -o "$CONTENTS/MacOS/airllm-desktop" ./cmd/airllm-desktop

cat > "$CONTENTS/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>ru</string>
    <key>CFBundleDisplayName</key>
    <string>AirLLM</string>
    <key>CFBundleExecutable</key>
    <string>airllm-desktop</string>
    <key>CFBundleIdentifier</key>
    <string>io.github.lutikk.airllm.desktop</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>AirLLM</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>0.1.0</string>
    <key>CFBundleVersion</key>
    <string>1</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.13</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSHumanReadableCopyright</key>
    <string>AirLLM contributors</string>
</dict>
</plist>
PLIST

printf '%s\n' "Готово: $APP_DIR"
