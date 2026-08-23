#!/usr/bin/env bash
set -u

signing_output="$(security find-identity -v -p codesigning 2>&1 || true)"
if printf '%s\n' "$signing_output" | grep -q 'Developer ID Application:'; then
  echo "[OK] Developer ID Application certificate found"
  signing_ok=0
else
  echo "[BLOCKED] Developer ID Application certificate not found"
  echo "[INFO] Apple Development certificates cannot be used for notarized distribution"
  signing_ok=1
fi

profile="${NOTARY_PROFILE:-}"
if [[ -z "$profile" ]]; then
  echo "[BLOCKED] NOTARY_PROFILE is not set"
  notary_ok=1
else
  if xcrun notarytool history --keychain-profile "$profile" >/dev/null 2>&1; then
    echo "[OK] notarization profile is usable: $profile"
    notary_ok=0
  else
    echo "[BLOCKED] notarization profile could not be used: $profile"
    notary_ok=1
  fi
fi

if [[ "$signing_ok" -eq 0 && "$notary_ok" -eq 0 ]]; then
  echo "[OK] release signing readiness"
  exit 0
fi
exit 1
