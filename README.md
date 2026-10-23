# Face Swap, Extend & Templates

One repository for the Face Swap RunPod worker, Wan 2.2 Extend Home Screen
launcher, and Wan 2.2 Templates web app. The web app has navigation between
all three tools. Each tool keeps its existing backend and input/output contract.

## Tools

| Tool | Web route | Source | Backend |
|------|-----------|--------|---------|
| Templates | `/` | `index.html`, `src/`, `server/`, `api/`, `netlify/` | Existing Wan Animate RunPod endpoint; HF Space for template Extend |
| Extend | `/extend/` | `public/extend/` | Opens `https://simzy-wan22-extend.hf.space/` |
| Face Swap | `/face-swap/` | `public/face-swap/` launcher; `handler.py` GPU worker | Opens the private `Simzy/face-swap-runpod` Space |

The standalone Extend generator and the private Face Swap UI are hosted on
Hugging Face; their application source was not in these GitHub repositories.
The Extend launcher keeps its icons, manifest, and Home Screen redirect.
The Templates Space proxy source is included in `space/`.

## Local web development

Use Node.js 22 or newer and npm:

```bash
npm ci
npm test
npm run dev -- --host 0.0.0.0
```

Open `http://localhost:5173/`. All three navigation routes work from the same
site. Templates API routes run through Vite's existing development middleware.
Face Swap and Extend open their hosted apps when you choose **Open**.

`npm run build` produces `dist/`, including the two launchers and Extend icons.
`npm run preview` serves the built static site; it does not run API handlers.

Generation needs the existing server environment variables described in
[Templates setup](docs/templates.md). Keep `RUNPOD_API_KEY` and `HF_TOKEN` on
the server. The Templates RunPod endpoint and the Face Swap worker are separate
models and deployments. Template Extend still cannot continue RunPod clips.

## Deploy the web app

Both `vercel.json` and `netlify.toml` are at the repository root. Configure the
hosting project to use this combined repository with root directory `.`:

- Build: `npm run build`
- Publish/output: `dist`
- Netlify functions: `netlify/functions`
- Vercel functions: `api/`

To migrate an existing Templates or Extend site, change its Git repository
connection to this repository, retain its server environment variables, and
redeploy. Existing sites remain on their current code until that migration.
An old Extend-only site's Home Screen installation must be added again from
`/extend/` on the combined site to use the new launcher location.

## Deploy the Face Swap worker

The root `Dockerfile`, `requirements.txt`, `handler.py`, and GitHub Actions
image workflow keep their existing paths. From the repository root:

```bash
docker build -t face-swap-runpod-worker .
```

Pushes to `main` still build `ghcr.io/simzy420/face-swap-runpod-worker:latest`.
The web build does not load model weights. Running the Python worker needs
Python 3.12, its CUDA dependencies, a 48 GB+ GPU, and the Qwen cached model
configuration described in [Face Swap setup](docs/face-swap.md).

## Further documentation

- [Templates setup, API behavior, and Space deployment](docs/templates.md)
- [Face Swap worker setup and model licenses](docs/face-swap.md)
- [Imported repositories and source revisions](docs/sources.md)

This consolidation does not merge or delete the original remote repositories
or transfer their Git history. Source revisions are recorded for traceability.
