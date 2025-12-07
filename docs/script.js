"use strict";

/** Global variable storing the fetched optimizer data. */
let globalData = null;

/**
 * Switch between ranking tabs.
 *
 * @param {Event} evt - The click event from the tab button.
 * @param {string} tabName - The ID of the tab content to display.
 */
function openTab(evt, tabName) {
  const tabContents = document.querySelectorAll(".tab-content");
  tabContents.forEach((content) => {
    content.style.display = "none";
  });

  const tabLinks = document.querySelectorAll(".tab-link");
  tabLinks.forEach((link) => {
    link.classList.remove("active");
  });

  const selectedTab = document.getElementById(tabName);
  if (selectedTab) {
    selectedTab.style.display = "block";
    selectedTab.style.opacity = 0;
    setTimeout(() => (selectedTab.style.opacity = 1), 50);
  }

  if (evt && evt.currentTarget) {
    evt.currentTarget.classList.add("active");
  }
}

/**
 * Escape special regex characters in a string.
 *
 * @param {string} string - The string to escape.
 * @returns {string} The escaped string safe for use in a RegExp.
 */
function escapeRegExp(string) {
  return string.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/**
 * Wrap matched text in a highlight span while preserving original casing.
 *
 * @param {string} text - The text to search within.
 * @param {string} searchTerm - The term to highlight.
 * @returns {string} HTML string with matches wrapped in highlight spans.
 */
function highlightText(text, searchTerm) {
  if (!searchTerm) return text;

  const escapedTerm = escapeRegExp(searchTerm);
  const regex = new RegExp(`(${escapedTerm})`, "gi");

  return text.replace(regex, '<span class="highlight">$1</span>');
}

/**
 * Populate a table body with optimizer ranking data.
 *
 * @param {HTMLElement} tableBody - The tbody element to populate.
 * @param {Array<Object>} data - Array of optimizer objects with rank, optimizer, value, and vis properties.
 * @param {string} [searchTerm=""] - Optional search term for highlighting matches.
 */
function populateTable(tableBody, data, searchTerm = "") {
  if (!tableBody || !data) {
    console.warn("Table body or data missing.");
    return;
  }

  if (data.length === 0) {
    tableBody.innerHTML = `<tr><td colspan="4" style="text-align:center; padding: 30px; color: var(--accents-4);">No optimizers found matching "${searchTerm}"</td></tr>`;
    return;
  }

  tableBody.innerHTML = "";

  const rows = data
    .map((item) => {
      const optimizerNameDisplay = highlightText(item.optimizer, searchTerm);

      return `
        <tr>
            <td class="rank-cell">
                <span style="font-weight:700; color:var(--accents-6);">#${item.rank}</span>
            </td>
            <td class="opt-name" style="font-weight:500;">
                ${optimizerNameDisplay}
            </td>
            <td class="score-cell">${item.value}</td>
            <td class="vis-link">
                <a href="${item.vis}" class="action-link vis-link-icon">
                    View Trajectory
                    <svg
                        width="14"
                        height="14"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        stroke-width="2"
                        stroke-linecap="round"
                        stroke-linejoin="round"
                    >
                        <path d="M5 12h14M12 5l7 7-7 7"/>
                    </svg>
                </a>
            </td>
        </tr>
    `;
    })
    .join("");

  tableBody.innerHTML = rows;
}

/**
 * Handle search input and filter both ranking tables.
 *
 * @param {Event} evt - The input event from the search field.
 */
function handleSearch(evt) {
  if (!globalData) return;

  const searchTerm = evt.target.value.toLowerCase().trim();
  const rawSearchTerm = evt.target.value.trim();

  const filteredRank = globalData.rankingByAvgRank.filter((item) =>
    item.optimizer.toLowerCase().includes(searchTerm),
  );
  const filteredError = globalData.rankingByErrorRate.filter((item) =>
    item.optimizer.toLowerCase().includes(searchTerm),
  );

  const rankingTableBody = document.getElementById("ranking-table-body");
  const errorRateTableBody = document.getElementById("error-rate-table-body");

  if (rankingTableBody)
    populateTable(rankingTableBody, filteredRank, rawSearchTerm);
  if (errorRateTableBody)
    populateTable(errorRateTableBody, filteredError, rawSearchTerm);

  const tableContainers = document.querySelectorAll(".table-container");
  tableContainers.forEach((container) => {
    container.scrollTop = 0;
  });
}

/**
 * Fetch ranking data and initialize the dashboard tables.
 */
async function loadDataAndInitializeTables() {
  try {
    const response = await fetch("ranks.json");

    if (!response.ok) {
      throw new Error(`Failed to load data: ${response.status}`);
    }

    globalData = await response.json();

    const rankingTableBody = document.getElementById("ranking-table-body");
    const errorRateTableBody = document.getElementById("error-rate-table-body");

    if (rankingTableBody)
      populateTable(rankingTableBody, globalData.rankingByAvgRank);
    if (errorRateTableBody)
      populateTable(errorRateTableBody, globalData.rankingByErrorRate);

    const searchInput = document.getElementById("optimizer-search");
    if (searchInput) {
      searchInput.addEventListener("input", handleSearch);
    }
  } catch (error) {
    console.error("Error initializing dashboard:", error);
  }
}

document.addEventListener("DOMContentLoaded", loadDataAndInitializeTables);
