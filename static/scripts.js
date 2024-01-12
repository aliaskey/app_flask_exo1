let canvas = document.getElementById('canvas');
let ctx = canvas.getContext('2d');
ctx.fillStyle = "white";
ctx.fillRect(0, 0, canvas.width, canvas.height);

let drawing = false;
let prevX = 0;
let prevY = 0;

canvas.addEventListener('mousedown', (e) => {
    drawing = true;
    prevX = e.clientX - canvas.offsetLeft;
    prevY = e.clientY - canvas.offsetTop;
});

canvas.addEventListener('mouseup', () => {
    drawing = false;
});

canvas.addEventListener('mousemove', (e) => {
    if (drawing) {
        let currentX = e.clientX - canvas.offsetLeft;
        let currentY = e.clientY - canvas.offsetTop;
        draw(prevX, prevY, currentX, currentY);
        prevX = currentX;
        prevY = currentY;
    }
});

function draw(x1, y1, x2, y2) {
    ctx.beginPath();
    ctx.strokeStyle = 'black';
    ctx.lineWidth = 2;
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
    ctx.closePath();
}

function clearCanvas() {
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}

function predictDigit() {
    let imageData = canvas.toDataURL('image/png');
    // Envoyer cette imageData à votre route Flask pour prédiction
}
