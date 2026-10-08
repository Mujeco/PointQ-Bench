"use strict";

const caseDescriptions = {
  pointcloud:
    "Rendered point cloud shown in the paper's quality-assessment example.",
  reference: "Original mesh: visual reference only, not an evaluation input.",
};
const caseButtons = document.querySelectorAll("[data-case-view]");
const caseImages = document.querySelectorAll("[data-case-image]");
caseButtons.forEach((button) => {
  button.addEventListener("click", () => {
    const view = button.dataset.caseView;
    if (!Object.hasOwn(caseDescriptions, view)) return;
    caseButtons.forEach((other) => {
      other.setAttribute("aria-pressed", String(other === button));
    });
    caseImages.forEach((img) => {
      img.hidden = img.dataset.caseImage !== view;
    });
    document.getElementById("case-description").textContent =
      caseDescriptions[view];
  });
});

let toastTimer;
async function copyText(text) {
  const toast = document.getElementById("copy-status");
  try {
    await navigator.clipboard.writeText(text);
    toast.textContent = "Copied to clipboard.";
  } catch {
    // Keep a usable fallback when browser clipboard permissions are denied.
    toast.textContent =
      "Copy unavailable. Select and copy the visible text instead.";
  }
  toast.classList.add("visible");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove("visible"), 3500);
}

document
  .querySelectorAll("[data-copy]")
  .forEach((button) =>
    button.addEventListener("click", () => copyText(button.dataset.copy)),
  );

document
  .getElementById("copy-citation")
  .addEventListener("click", () =>
    copyText(document.getElementById("bibtex").textContent),
  );
