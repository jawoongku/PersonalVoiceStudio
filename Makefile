.PHONY: verify app ui

verify:
	scripts/verify_all.sh

app:
	scripts/build_macos_app.sh

ui:
	conda run -n cosyvoice python -m mac_voice ui
