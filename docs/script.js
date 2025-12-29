"use strict";

/**
 * Global state for storing fetched benchmark data.
 * @type {{ rankingByAvgRank: Array, rankingByWeightedRank: Array } | null}
 */
let benchmarkData = null;

/**
 * DOM Elements cache to avoid repeated lookups.
 */
const elements = {
  loader: document.getElementById("loader-container"),
  rankingsSection: document.getElementById("rankings"),
  avgRankTable: document.getElementById("ranking-table-body"),
  weightedRankTable: document.getElementById("weighted-rank-table-body"),
  searchInput: document.getElementById("optimizer-search"),
  tabLinks: document.querySelectorAll(".tab-link"),
  tabContents: document.querySelectorAll(".tab-content"),
};

/**
 * Switch between ranking tabs (Weighted Rank vs Average Rank).
 * @param {Event} evt - The click event.
 * @param {string} targetTabId - The ID of the content div to show.
 */
function openTab(evt, targetTabId) {
  // Hide all contents
  elements.tabContents.forEach((content) => {
    content.style.display = "none";
  });

  // Deactivate all buttons
  elements.tabLinks.forEach((link) => {
    link.classList.remove("active");
  });

  // Show target content
  const selectedContent = document.getElementById(targetTabId);
  if (selectedContent) {
    selectedContent.style.display = "block";
    // Trigger tiny fade-in effect
    selectedContent.style.opacity = 0;
    requestAnimationFrame(() => {
      selectedContent.style.opacity = 1;
    });
  }

  // Activate clicked button
  if (evt && evt.currentTarget) {
    evt.currentTarget.classList.add("active");
  }
}

/**
 * Utility: Escape special characters for RegExp.
 */
function escapeRegExp(string) {
  return string.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/**
 * Utility: Highlight matching text in strings.
 */
function highlightMatch(text, searchTerm) {
  if (!searchTerm) return text;
  const regex = new RegExp(`(${escapeRegExp(searchTerm)})`, "gi");
  return text.replace(regex, '<span class="highlight">$1</span>');
}

/**
 * Renders data into a specific table body.
 * @param {HTMLElement} tableBody - The tbody element.
 * @param {Array} data - The array of optimizer objects.
 * @param {string} searchTerm - Current search string for highlighting.
 */
function renderTable(tableBody, data, searchTerm = "") {
  if (!tableBody) return;

  if (!data || data.length === 0) {
    tableBody.innerHTML = `
      <tr>
        <td colspan="4" style="text-align:center; padding: 2rem; color: var(--accents-4);">
          No optimizers found matching "${searchTerm}"
        </td>
      </tr>`;
    return;
  }

  // Use a document fragment for better performance on large lists
  const fragment = document.createDocumentFragment();

  data.forEach((item) => {
    const tr = document.createElement("tr");

    // Highlight name if searching
    const displayName = highlightMatch(item.optimizer, searchTerm);

    tr.innerHTML = `
      <td class="rank-cell">
        <span style="font-weight:700;">#${item.rank}</span>
      </td>
      <td class="opt-name">
        ${displayName}
      </td>
      <td class="score-cell">${item.value}</td>
      <td class="vis-link">
        <a href="${item.vis}" class="action-link vis-link-icon">
          View Trajectory
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M5 12h14M12 5l7 7-7 7"/>
          </svg>
        </a>
      </td>
    `;
    fragment.appendChild(tr);
  });

  tableBody.innerHTML = "";
  tableBody.appendChild(fragment);
}

/**
 * Filter data based on search input and re-render active tables.
 * @param {string} query - The search string.
 */
function filterAndRender(query) {
  if (!benchmarkData) return;

  const lowerQuery = query.toLowerCase();

  // Filter both datasets
  const filteredAvg = benchmarkData.rankingByAvgRank.filter((item) =>
    item.optimizer.toLowerCase().includes(lowerQuery),
  );

  const filteredWeighted = benchmarkData.rankingByWeightedRank.filter((item) =>
    item.optimizer.toLowerCase().includes(lowerQuery),
  );

  // Update DOM
  renderTable(elements.avgRankTable, filteredAvg, query);
  renderTable(elements.weightedRankTable, filteredWeighted, query);
}

/**
 * Setup event listeners.
 */
function setupEventListeners() {
  if (elements.searchInput) {
    elements.searchInput.addEventListener("input", (e) => {
      filterAndRender(e.target.value.trim());
    });
  }
}

/**
 * Main initialization function.
 * Fetches data, handles loading state, and initial render.
 */
async function initDashboard() {
  try {
    // 1. Minimum delay to prevent flickering loader on fast networks (UX polish)
    const minDelay = new Promise((resolve) => setTimeout(resolve, 600));

    // 2. Fetch Data
    const fetchRequest = fetch("ranks.json").then((res) => {
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return res.json();
    });

    const [_, data] = await Promise.all([minDelay, fetchRequest]);
    benchmarkData = data;

    // 3. Initial Render
    filterAndRender("");

    // 4. Hide Loader & Show Content
    if (elements.loader) elements.loader.style.display = "none";
    if (elements.rankingsSection) {
      elements.rankingsSection.style.display = "block";
      // Trigger fade-in
      setTimeout(() => {
        elements.rankingsSection.style.opacity = 1;
      }, 50);
    }

    // 5. Setup interactions
    setupEventListeners();

    // 6. Bind global tab function for HTML onclick attributes
    window.openTab = openTab;
  } catch (err) {
    console.error("Dashboard Init Error:", err);
    if (elements.loader) {
      elements.loader.innerHTML = `
        <p style="color: #d32f2f; font-weight: 500;">
          Failed to load benchmark data.
        </p>
        <button onclick="location.reload()" style="margin-top:1rem; padding:8px 16px; cursor:pointer;">
          Retry
        </button>
      `;
    }
  }
}

// Start application when DOM is ready
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initDashboard);
} else {
  initDashboard();
}
