document.addEventListener('DOMContentLoaded', (event) => {
    let draggables = document.querySelectorAll('.draggable-text');
    let textContainer = document.getElementById('text-container');
    
    draggables.forEach((draggable, index) => {
        draggable.style.position = 'absolute';
        draggable.style.left = `${(index % 6) * 16}%`;
        draggable.style.top = `${Math.floor(index / 6) * 40}px`;
        
        draggable.addEventListener('mousedown', dragStart);
        
        function dragStart(e) {
            e.preventDefault();
            let startX = e.clientX - draggable.offsetLeft;
            let startY = e.clientY - draggable.offsetTop;
            
            function dragMove(e) {
                let newX = e.clientX - startX;
                let newY = e.clientY - startY;
                
                newX = Math.max(0, Math.min(newX, textContainer.offsetWidth - draggable.offsetWidth));
                newY = Math.max(0, Math.min(newY, textContainer.offsetHeight - draggable.offsetHeight));
                
                draggable.style.left = `${newX}px`;
                draggable.style.top = `${newY}px`;
            }
            
            function dragEnd() {
                document.removeEventListener('mousemove', dragMove);
                document.removeEventListener('mouseup', dragEnd);
            }
            
            document.addEventListener('mousemove', dragMove);
            document.addEventListener('mouseup', dragEnd);
        }
    });
});