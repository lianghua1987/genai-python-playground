const form = document.querySelector("#form");

const createColorBoxes = (colors, parentElement) => {
  parentElement.innerHTML = "";
  Array.from(colors).forEach(color => {
    const div = document.createElement("div");
    div.style.backgroundColor = color;
    div.style.width = `calc(100% / ${colors.length})`;
    div.classList.add("color");
    div.addEventListener("click", function () {
      navigator.clipboard.writeText(color);
    });
    const span = document.createElement("span");
    span.innerHTML = color;
    div.appendChild(span);
    parentElement.appendChild(div);
  });
};

const getColors = () => {
  const searchQuery = form.elements.query.value;
    fetch("/palette", {
      method: "POST",
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({query: searchQuery})
    })
    .then(response => response.json())
    .then(data => {
      const colors = data.colors;
      const container = document.querySelector(".container");
      createColorBoxes(colors, container);
    });
};

form.addEventListener("submit", function (e) {
  e.preventDefault();
  getColors();
});