function interpolateColor(color1, color2, factor) {
    if (factor === undefined) factor = 0.5;

    let result = color1.slice(1).match(/.{2}/g)
        .map((hex, i) => {
            return Math.round(
                parseInt(hex, 16) * (1 - factor) +
                parseInt(color2.slice(1).match(/.{2}/g)[i], 16) * factor
            );
        });

    return `#${result.map(value => {
        let hex = value.toString(16);
        return hex.length === 1 ? "0" + hex : hex;
    }).join('')}`;
}

function generateColorList(startColor, endColor, steps) {
    let colorList = [];
    for (let i = 0; i < steps; i++) {
        let factor = i / (steps - 1); // Interpolation factor from 0 to 1
        colorList.push(interpolateColor(startColor, endColor, factor));
    }
    return colorList;
}

// Generar 100 colores
let colours = generateColorList("#FFFF3F", "#007F5F", 101);
var values = [];

var tds = document.getElementsByTagName("td");
for (let i = 0; i < tds.length; i++) {
    let td = tds[i];
    let value = td.textContent;

    if (value.startsWith("u") || value.startsWith("r")) {
        continue;
    }

    td.style.backgroundColor = colours[td.textContent];
}

