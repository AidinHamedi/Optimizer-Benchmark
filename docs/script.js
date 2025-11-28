"use strict";

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
 * Populates a table body
 */
function populateTable(tableBody, data) {
  if (!tableBody || !data) {
    console.warn("Table body or data missing.");
    return;
  }

  tableBody.innerHTML = "";

  const rows = data
    .map(
      (item) => `
        <tr>
            <td class="rank-cell">
                <span style="font-weight:700; color:var(--accents-6);">#${item.rank}</span>
            </td>
            <td class="opt-name" style="font-weight:500;">${item.optimizer}</td>
            <td class="score-cell">${item.value}</td>
            <td class="vis-link">
                <!-- Added 'action-link' class here -->
                <a href="${item.vis}" class="action-link">
                    View Analysis
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="margin-left:6px;"><path d="M5 12h14M12 5l7 7-7 7"/></svg>
                </a>
            </td>
        </tr>
    `,
    )
    .join("");

  tableBody.innerHTML = rows;
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

    const data = await response.json();

    const rankingTableBody = document.getElementById("ranking-table-body");
    const errorRateTableBody = document.getElementById("error-rate-table-body");

    if (rankingTableBody)
      populateTable(rankingTableBody, data.rankingByAvgRank);
    if (errorRateTableBody)
      populateTable(errorRateTableBody, data.rankingByErrorRate);
  } catch (error) {
    console.error("Error initializing dashboard:", error);
  }
}

document.addEventListener("DOMContentLoaded", loadDataAndInitializeTables);
