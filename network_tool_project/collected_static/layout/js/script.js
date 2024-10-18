
// layout/static/layout/js/script.js

const canvas = new fabric.Canvas('network');

function addNode(type, labelText, left = '100px', top = '100px') {
    const node = new fabric.Rect({
        left: parseInt(left),
        top: parseInt(top),
        fill: (type === 'substation') ? 'red' : 'blue',
        width: 20,
        height: 20,
        hasControls: false,
        hasBorders: false
    });
    node.labelText = labelText;
    canvas.add(node);
}

function addTransformer(type, labelText, left = '100px', top = '100px') {
    addNode('transformer', labelText, left, top);
}

function setDrawMode(type) {
    // Implement draw mode functionality
}

function setSelectionMode() {
    // Implement selection mode functionality
}

function deleteSelected() {
    const activeObject = canvas.getActiveObject();
    if (activeObject) {
        canvas.remove(activeObject);
    }
}

function saveLayout() {
    const layout = JSON.stringify(canvas.toJSON());
    document.getElementById('output').value = layout;
}

function loadLayout(input) {
    const reader = new FileReader();
    reader.onload = function(event) {
        const json = event.target.result;
        canvas.loadFromJSON(json);
    };
    reader.readAsText(input.files[0]);
}

function downloadLayout() {
    const layout = document.getElementById('output').value;
    const blob = new Blob([layout], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'network_layout.json';
    a.click();
    URL.revokeObjectURL(url);
}

function addCustomText(defaultText = '', left = '100px', top = '100px') {
    const text = new fabric.Textbox(defaultText, {
        left: parseInt(left),
        top: parseInt(top),
        width: 150,
        fontSize: 20,
        fill: 'black',
        editable: true
    });
    canvas.add(text);
}

document.getElementById('network').addEventListener('wheel', (event) => {
    event.preventDefault();
    const zoom = canvas.getZoom();
    const delta = event.deltaY;
    const newZoom = zoom + delta / 1000;
    canvas.setZoom(newZoom);
});

document.getElementById('network').addEventListener('mousemove', (event) => {
    if (event.buttons === 1) {
        const moveX = event.movementX;
        const moveY = event.movementY;
        canvas.relativePan({ x: moveX, y: moveY });
    }
});
