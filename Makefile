.PHONY: verify app check-app package-release ui notarize mps-doctor create-mps-env release-readiness

verify:
	scripts/verify_all.sh

app:
	scripts/build_macos_app.sh

check-app:
	scripts/check_app_bundle.sh

package-release:
	scripts/package_macos_release.sh

ui:
	conda run -n cosyvoice python -m mac_voice ui

mps-doctor:
	conda run -n cosyvoice python -m mac_voice mps-doctor

create-mps-env:
	scripts/create_mps_env.sh --dry-run

release-readiness:
	scripts/check_release_readiness.sh

notarize:
	scripts/notarize_macos_app.sh
