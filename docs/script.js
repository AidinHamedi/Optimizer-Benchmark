function openTab(evt, tabName) {
  const tabcontent = document.getElementsByClassName("tab-content");
  for (let i = 0; i < tabcontent.length; i++) {
    tabcontent[i].style.display = "none";
  }

  const tablinks = document.getElementsByClassName("tab-link");
  for (let i = 0; i < tablinks.length; i++) {
    tablinks[i].className = tablinks[i].className.replace(" active", "");
  }

  document.getElementById(tabName).style.display = "block";
  evt.currentTarget.className += " active";
}

function populateTable(tableBody, data) {
  tableBody.innerHTML = "";
  if (!data) {
    console.error("Data for table is undefined.");
    return;
  }
  data.forEach((item) => {
    const row = document.createElement("tr");
    row.innerHTML = `
            <td>${item.rank}</td>
            <td>${item.optimizer}</td>
            <td>${item.value}</td>
            <td class="vis-link"><a href="${item.vis}">Open</a></td>
        `;
    tableBody.appendChild(row);
  });
}

async function loadDataAndInitializeTables() {
  try {
    const response = await fetch("ranks.json");
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const data = await response.json();

    const rankingTableBody = document.getElementById("ranking-table-body");
    const errorRateTableBody = document.getElementById("error-rate-table-body");

    populateTable(rankingTableBody, data.rankingByAvgRank);
    populateTable(errorRateTableBody, data.rankingByErrorRate);
  } catch (error) {
    console.error("Failed to load or process benchmark data:", error);
  }
}

document.addEventListener("DOMContentLoaded", loadDataAndInitializeTables);
