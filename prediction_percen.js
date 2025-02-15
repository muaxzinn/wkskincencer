


//#############################################################

// ### 1. LOAD THE MODEL IMMEDIATELY WHEN THE PAGE LOADS

//#############################################################


// Define 2 helper functions

function simulateClick(tabID) {

        document.getElementById(tabID).click();
}



function predictOnLoad() {

        // Simulate a click on the predict button
        setTimeout(simulateClick.bind(null,'predict-button'), 500);
}






// LOAD THE MODEL

let model;
(async function () {

        model = await tf.loadModel('https://muaxzinn.github.io/wkskincencer/model/model.json'); //เปลี่ยนตรงนี้ 
        $("#selected-image").attr("src", "https://muaxzinn.github.io/wkskincencer/assets/samplepic.jpg"); // กับตรงนี้ 

        // Hide the model loading spinner
        // This line of html gets hidden:
        // <div class="progress-bar">Ai is Loading...</div>
        $('.progress-bar').hide();


        // Simulate a click on the predict button.
        // Make a prediction on the default front page image.
        predictOnLoad();



})();





$("#predict-button").click(async function () {

    let image = undefined;

    image = $('#selected-image').get(0);

    // Pre-process the image
    let tensor = tf.fromPixels(image)
        .resizeNearestNeighbor([224, 224])
        .toFloat();

    let offset = tf.scalar(127.5);

    tensor = tensor.sub(offset)
        .div(offset)
        .expandDims();

    // Predict using the model
    let predictions = await model.predict(tensor).data();
    let top5 = Array.from(predictions)
        .map(function (p, i) { // this is Array.map
            return {
                probability: p * 100,  // แปลงเป็นเปอร์เซ็นต์
                className: TARGET_CLASSES[i]
            };
        }).sort(function (a, b) {
            return b.probability - a.probability;
        }).slice(0, 3);

    // ล้างผลลัพธ์เก่าทั้งหมด
    $("#prediction-list").empty();

    // Append the file name to the prediction list
    var file_name = 'samplepic.jpg';
    $("#prediction-list").append(`<li class="w3-text-blue fname-font" style="list-style-type:none;">${file_name}</li>`);

    // แสดงผลการทำนายในรูปแบบเปอร์เซ็นต์
    top5.forEach(function (p) {
        $("#prediction-list").append(`<li style="list-style-type:none;">${p.className}: ${p.probability.toFixed(3)}%</li>`);  // แสดงเป็นเปอร์เซ็นต์
    });
});

//######################################################################

// ### 3. READ THE IMAGES THAT THE USER SELECTS

// Then direct the code execution to app_batch_prediction_code.js

//######################################################################




// This listens for a change. It fires when the user submits images.

$("#image-selector").change(async function () {

        // the FileReader reads one image at a time
        fileList = $("#image-selector").prop('files');

        //$("#prediction-list").empty();

        // Start predicting
        // This function is in the app_batch_prediction_code.js file.
        model_processArray(fileList);

});



