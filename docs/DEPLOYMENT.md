# Static Project Page

This page uses plain HTML, CSS, JavaScript, and a self-hosted OFL font. It
has no analytics, third-party script requests, API backend, or build step.
`docs/.nojekyll` disables Jekyll processing.

## Local Preview

From the repository root:

```sh
python -m http.server 8000 --bind 127.0.0.1 --directory docs
```

Visit `http://127.0.0.1:8000/`. The data mirror is already public; local
preview does not publish the website or the repository changes.

## GitHub Pages

After the author approves the exact release contents, push them to
`Mujeco/PointQ-Bench`. In the repository's **Settings > Pages**, select
**Deploy from a branch**, branch **main**, folder **/docs**, then save.

The expected project URL is `https://mujeco.github.io/PointQ-Bench/`.
An expected URL is not evidence of deployment: check the Pages build status
and open the published URL before announcing it.

GitHub supports publishing static files from a branch's `docs` directory:
[GitHub Pages documentation](https://docs.github.com/en/pages/getting-started-with-github-pages/creating-a-github-pages-site).

Alternatively, dispatch the optional `pages.yml` workflow after configuring
the repository's Pages source as GitHub Actions. The workflow publishes
only `docs/`, not datasets, logs, or internal audit material.

## Maintenance

- Keep the Baidu link, code, and package counts synchronized in `index.html`,
  the repository README, `data/README.md`, and `data/release_manifest.json`.
- Never call the illustrative chicken a benchmark sample or judge result.
- Pair formal model/reference bundles before adding numerical leaderboards.
- The paper figure and PDF are separate from MIT code; preserve notices.
- `404.html`, Open Graph metadata, robots, and sitemap assume this exact
  GitHub Pages project prefix. Update them if changing the hosting URL.
