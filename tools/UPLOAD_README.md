# Drive uploads — one-time rclone setup

The project uploads built results (HTML presentations, plots, npz
artifacts) to a shared Google Drive folder named **Quaia** inside
`My Drive/tmp/`, owned by `tabel@stanford.edu`. View at:

  https://drive.google.com/drive/folders/16FquBrDQdwxFArqwr35uoKHuAr5e_aRA

This is the **default destination for all results** going forward.
The build scripts call `tools/drive_upload.sh` automatically when
`PAPER_DRIVE_UPLOAD=1` is set.

## One-time rclone setup (~2 minutes)

```bash
# 1. Install rclone if not already (Homebrew on macOS):
brew install rclone

# 2. Run the interactive configurator. Answer the prompts:
rclone config
```

At the prompts:

| Prompt | Answer |
|---|---|
| n) New remote / s) Set configuration password ... | `n` |
| name> | `gdrive` |
| Storage> | `drive` (Google Drive) |
| client_id> | *(leave blank — uses rclone's default OAuth app)* |
| client_secret> | *(blank)* |
| scope> | `1` (full access) |
| service_account_file> | *(blank)* |
| Edit advanced config> | `n` |
| Use auto config> | `y` *(opens browser, sign in as `tabel@stanford.edu`, click "Allow")* |
| Configure this as a Shared Drive> | `n` |
| Yes this is OK> | `y` (confirm config) |
| Quit> | `q` |

That's it. Verify with:

```bash
rclone listremotes
# Expected output: gdrive:

rclone ls gdrive:tmp/Quaia | head
# Expected: a list of files already in the Quaia folder
```

## Daily use

Once configured, the build scripts upload automatically:

```bash
# Default-on for all rebuilds (recommended):
export PAPER_DRIVE_UPLOAD=1

python demos/build_knn_cdf_desi_quaia_presentation.py
# → builds the HTML and uploads it to Quaia

# Manually upload an arbitrary file:
tools/drive_upload.sh output/some_plot.png

# Override destination if needed:
PAPER_DRIVE_PATH="tmp/Quaia/sub" tools/drive_upload.sh output/foo.npz
```

## Environment variables

- `PAPER_DRIVE_UPLOAD` — set to `1` to enable post-build uploads in
  `demos/build_knn_cdf_desi_quaia_presentation.py`.
- `PAPER_DRIVE_REMOTE` — rclone remote name (default `gdrive`).
- `PAPER_DRIVE_PATH` — destination path on the remote (default
  `tmp/Quaia` — the project's results folder).
