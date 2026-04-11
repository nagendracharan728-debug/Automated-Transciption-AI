const fileInput = document.getElementById("audioFile")
const audioPlayer = document.getElementById("audioPlayer")
const captionBox = document.getElementById("captions")

const playBtn = document.getElementById("playBtn")
const stopBtn = document.getElementById("stopBtn")

const captions = [
{time:1,text:"Welcome to the AI podcast analyzer"},
{time:4,text:"Today we talk about artificial intelligence"},
{time:7,text:"Machine learning is transforming industries"},
{time:10,text:"Data and algorithms power modern AI systems"},
{time:13,text:"Thank you for listening to this podcast"}
]

let currentCaption = 0

fileInput.addEventListener("change",(e)=>{

const file = e.target.files[0]

if(file){

const url = URL.createObjectURL(file)

audioPlayer.src = url

currentCaption = 0

captionBox.innerHTML = "<p>Ready to play...</p>"

}

})

playBtn.onclick = () => {

audioPlayer.play()

}

stopBtn.onclick = () => {

audioPlayer.pause()

audioPlayer.currentTime = 0

currentCaption = 0

captionBox.innerHTML = "<p>Audio stopped</p>"

}

setInterval(()=>{

if(!audioPlayer.paused){

const currentTime = audioPlayer.currentTime

if(currentCaption < captions.length){

if(currentTime >= captions[currentCaption].time){

captionBox.innerHTML =
"<p class='caption-line'>" + captions[currentCaption].text + "</p>"

currentCaption++

}

}

}

},300)
const startBtn = document.getElementById("start-btn");
const captionsDiv = document.getElementById("captions");
const audioFileInput = document.getElementById("audio-file");

startBtn.addEventListener("click", async () => {
    const file = audioFileInput.files[0];
    if (!file) return alert("Upload audio first!");

    captionsDiv.innerHTML = "Transcribing...";

    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch("/transcribe", {
            method: "POST",
            body: formData
        });
        const data = await response.json();

        if (data.error) {
            captionsDiv.innerHTML = "Error: " + data.error;
        } else {
            captionsDiv.innerHTML = data.text;
        }
    } catch (err) {
        captionsDiv.innerHTML = "Error: " + err;
    }
});