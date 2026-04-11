const fileInput = document.getElementById("fileInput")

const wavesurfer = WaveSurfer.create({

container:"#waveform",
waveColor:"#6366f1",
progressColor:"#22d3ee",
height:120

})

fileInput.addEventListener("change",(e)=>{

const file = e.target.files[0]

const url = URL.createObjectURL(file)

wavesurfer.load(url)

})
