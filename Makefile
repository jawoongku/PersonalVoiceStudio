.PHONY: verify app ui notarize

verify:
	scripts/verify_all.sh

app:
	scripts/build_macos_app.sh

ui:
	conda run -n cosyvoice python -m mac_voice ui

notarize:
	scripts/notarize_macos_app.sh
