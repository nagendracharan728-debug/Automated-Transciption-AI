particlesJS("particles-js",{
particles:{
number:{value:120},
size:{value:3},
move:{speed:1},
line_linked:{enable:true}
}
})

const fileInput = document.getElementById("fileInput")

if(fileInput){

const wavesurfer = WaveSurfer.create({
container:'#waveform',
waveColor:'#6366f1',
progressColor:'#22d3ee',
height:120
})

fileInput.addEventListener("change",(e)=>{
const file = e.target.files[0]
const url = URL.createObjectURL(file)
wavesurfer.load(url)
})

}

const sentimentChart = document.getElementById("sentimentChart")

if(sentimentChart){

new Chart(sentimentChart,{
type:'bar',
data:{
labels:['Positive','Neutral','Negative'],
datasets:[{
label:'Sentiment',
data:[10,5,2],
backgroundColor:['#22c55e','#facc15','#ef4444']
}]
}
})

}

const segmentChart = document.getElementById("segmentChart")

if(segmentChart){

new Chart(segmentChart,{
type:'pie',
data:{
labels:['Segment1','Segment2','Segment3'],
datasets:[{
data:[4,6,3],
backgroundColor:['#6366f1','#06b6d4','#ec4899']
}]
}
})

}

const cloud = document.getElementById("keywordCloud")

if(cloud){

WordCloud(cloud,{
list:[
["AI",40],
["Podcast",30],
["Technology",25],
["Machine Learning",20],
["Innovation",18]
],
gridSize:8,
weightFactor:5,
color:"random-dark",
rotateRatio:0.5
})

}
const canvas = document.getElementById("bg3d")

if(canvas){

const scene = new THREE.Scene()

const camera = new THREE.PerspectiveCamera(
75,
window.innerWidth/window.innerHeight,
0.1,
1000
)

const renderer = new THREE.WebGLRenderer({
canvas:canvas,
alpha:true
})

renderer.setSize(window.innerWidth,window.innerHeight)

const geometry = new THREE.IcosahedronGeometry(2,1)

const material = new THREE.MeshBasicMaterial({
color:0x6366f1,
wireframe:true
})

const sphere = new THREE.Mesh(geometry,material)

scene.add(sphere)

camera.position.z = 5

function animate(){

requestAnimationFrame(animate)

sphere.rotation.x += 0.003
sphere.rotation.y += 0.004

renderer.render(scene,camera)

}

animate()

}
const audioContext = new AudioContext()

const canvasVis = document.getElementById("visualizer")

if(canvasVis){

const ctx = canvasVis.getContext("2d")

navigator.mediaDevices.getUserMedia({audio:true})
.then(stream=>{

const analyser = audioContext.createAnalyser()

const source = audioContext.createMediaStreamSource(stream)

source.connect(analyser)

analyser.fftSize = 256

const bufferLength = analyser.frequencyBinCount

const dataArray = new Uint8Array(bufferLength)

function draw(){

requestAnimationFrame(draw)

analyser.getByteFrequencyData(dataArray)

ctx.clearRect(0,0,canvasVis.width,canvasVis.height)

for(let i=0;i<bufferLength;i++){

const barHeight = dataArray[i]

ctx.fillStyle="#6366f1"

ctx.fillRect(i*5,canvasVis.height-barHeight,4,barHeight)

}

}

draw()

})

}
document.addEventListener("mousemove",e=>{

const particles=document.querySelectorAll(".particle")

particles.forEach(p=>{

const speed=p.getAttribute("data-speed")

const x=(window.innerWidth-e.pageX*speed)/100
const y=(window.innerHeight-e.pageY*speed)/100

p.style.transform=`translateX(${x}px) translateY(${y}px)`

})

})
const socket = io();

// Elements
const audioFileInput = document.getElementById("audio-file");
const startBtn = document.getElementById("start-btn");
const captionsDiv = document.getElementById("captions");
const playBtn = document.getElementById("play-btn");
const pauseBtn = document.getElementById("pause-btn");
const stopBtn = document.getElementById("stop-btn");

let wavesurfer;

// Initialize Wavesurfer
function initWaveSurfer(file) {
    if (wavesurfer) wavesurfer.destroy();

    wavesurfer = WaveSurfer.create({
        container: "#waveform",
        waveColor: "#6366f1",
        progressColor: "#06b6d4",
        height: 128,
        responsive: true
    });

    const objectUrl = URL.createObjectURL(file);
    wavesurfer.load(objectUrl);

    playBtn.onclick = () => wavesurfer.play();
    pauseBtn.onclick = () => wavesurfer.pause();
    stopBtn.onclick = () => wavesurfer.stop();
}

// Upload + start transcription
startBtn.addEventListener("click", async () => {
    const file = audioFileInput.files[0];
    if (!file) return alert("Upload audio first!");

    captionsDiv.innerHTML = "";
    initWaveSurfer(file);

    // Upload file
    const formData = new FormData();
    formData.append("file", file);
    const uploadRes = await fetch("/upload_temp_audio", { method: "POST", body: formData });
    const uploadData = await uploadRes.json();
    if (uploadData.status !== "ok") return alert("Upload failed");

    // Start real-time transcription
    socket.emit("start_transcription", { filename: file.name });
});

// Receive chunked captions
socket.on("transcription_chunk", (data) => {
    const p = document.createElement("p");
    p.textContent = data.text;
    captionsDiv.appendChild(p);
    captionsDiv.scrollTop = captionsDiv.scrollHeight;
});

// Transcription done
socket.on("transcription_done", (data) => {
    const p = document.createElement("p");
    p.style.color = "#06b6d4";
    p.textContent = data.text;
    captionsDiv.appendChild(p);
});