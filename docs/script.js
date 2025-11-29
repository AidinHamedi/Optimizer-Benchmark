"use strict";

// Global variable to store the fetched data
let globalData = null;

/**
 * Handles Tab Switching
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
 * Utility: Escapes special characters for Regex
 * Ensures that searching for things like "+" doesn't break the regex
 */
function escapeRegExp(string) {
  return string.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/**
 * Utility: Wraps matched text in a highlight span
 * Preserves the original casing of the text
 */
function highlightText(text, searchTerm) {
  if (!searchTerm) return text;

  const escapedTerm = escapeRegExp(searchTerm);
  // 'gi' = global match, case insensitive
  const regex = new RegExp(`(${escapedTerm})`, "gi");

  return text.replace(regex, '<span class="highlight">$1</span>');
}

/**
 * Populates a table body
 * @param {HTMLElement} tableBody - The tbody element
 * @param {Array} data - Array of optimizer objects
 * @param {String} searchTerm - The current search text (for highlighting)
 */
function populateTable(tableBody, data, searchTerm = "") {
  if (!tableBody || !data) {
    console.warn("Table body or data missing.");
    return;
  }

  // If no data matches filter
  if (data.length === 0) {
    tableBody.innerHTML = `<tr><td colspan="4" style="text-align:center; padding: 30px; color: var(--accents-4);">No optimizers found matching "${searchTerm}"</td></tr>`;
    return;
  }

  tableBody.innerHTML = "";

  const rows = data
    .map((item) => {
      // Apply highlighting to the optimizer name
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
 * Filter data based on search input
 */
function handleSearch(evt) {
  if (!globalData) return;

  const searchTerm = evt.target.value.toLowerCase().trim();
  const rawSearchTerm = evt.target.value.trim();

  // Filter both lists
  const filteredRank = globalData.rankingByAvgRank.filter((item) =>
    item.optimizer.toLowerCase().includes(searchTerm),
  );
  const filteredError = globalData.rankingByErrorRate.filter((item) =>
    item.optimizer.toLowerCase().includes(searchTerm),
  );

  // Re-populate both tables
  const rankingTableBody = document.getElementById("ranking-table-body");
  const errorRateTableBody = document.getElementById("error-rate-table-body");

  if (rankingTableBody)
    populateTable(rankingTableBody, filteredRank, rawSearchTerm);
  if (errorRateTableBody)
    populateTable(errorRateTableBody, filteredError, rawSearchTerm);

  // NEW: Scroll both table containers to the top
  const tableContainers = document.querySelectorAll(".table-container");
  tableContainers.forEach((container) => {
    container.scrollTop = 0;
  });
}

/**
 * Main initialization
 */
async function loadDataAndInitializeTables() {
  try {
    const response = await fetch("ranks.json");

    if (!response.ok) {
      throw new Error(`Failed to load data: ${response.status}`);
    }

    // Store data globally
    globalData = await response.json();

    const rankingTableBody = document.getElementById("ranking-table-body");
    const errorRateTableBody = document.getElementById("error-rate-table-body");

    // Initial Population
    if (rankingTableBody)
      populateTable(rankingTableBody, globalData.rankingByAvgRank);
    if (errorRateTableBody)
      populateTable(errorRateTableBody, globalData.rankingByErrorRate);

    // Attach Search Listener
    const searchInput = document.getElementById("optimizer-search");
    if (searchInput) {
      searchInput.addEventListener("input", handleSearch);
    }
  } catch (error) {
    console.error("Error initializing dashboard:", error);
  }
}

document.addEventListener("DOMContentLoaded", loadDataAndInitializeTables);
