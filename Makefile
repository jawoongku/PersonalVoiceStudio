.PHONY: verify app check-app ui notarize mps-doctor

verify:
	scripts/verify_all.sh

app:
	scripts/build_macos_app.sh

check-app:
	scripts/check_app_bundle.sh

ui:
	conda run -n cosyvoice python -m mac_voice ui

mps-doctor:
	conda run -n cosyvoice python -m mac_voice mps-doctor

notarize:
	scripts/notarize_macos_app.sh
