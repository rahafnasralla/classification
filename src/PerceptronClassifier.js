import React, { useState, useEffect, useRef } from 'react';

// Perceptron Learning Algorithm
function perceptronLearning(data, labels, learningRate, maxIterations) {
  // Initialize weights and bias
  let weights = [0, 0];
  let bias = 0;

  // Iteration counter
  let iteration = 0;

  // Main loop
  while (iteration < maxIterations) {
    let errorCount = 0;

    // Iterate over each data point
    for (let i = 0; i < data.length; i++) {
      const prediction = predict(data[i], weights, bias);
      const actual = labels[i];

      // Update weights and bias if prediction is incorrect
      if (prediction !== actual) {
        for (let j = 0; j < weights.length; j++) {
          weights[j] += learningRate * actual * data[i][j];
        }
        bias += learningRate * actual;
        errorCount++;
      }
    }

    // If there are no errors, exit the loop
    if (errorCount === 0) {
      break;
    }

    iteration++;
  }

  return { weights, bias };
}

// Prediction function
function predict(input, weights, bias) {
  const activation = input.reduce((sum, value, index) => sum + value * weights[index], 0) + bias;
  return activation >= 0 ? 1 : -1;
}

function perceptronLearningOneVsAll(data, labels, learningRate, maxIterations) {
  const uniqueLabels = [...new Set(labels)];
  const classifiers = [];

  // Train a perceptron for each class
  for (let i = 0; i < uniqueLabels.length; i++) {
    const currentLabel = uniqueLabels[i];
    const binaryLabels = labels.map((label) => (label === currentLabel ? 1 : -1));
    const { weights, bias } = perceptronLearning(data, binaryLabels, learningRate, maxIterations);
    classifiers.push({ label: currentLabel, weights, bias });
  }

  return classifiers;
}



const PerceptronClassifier = () => {
  const canvasRef = useRef(null);
  const [dataPoints, setDataPoints] = useState([]);
  const [learningRate, setLearningRate] = useState(0.1);
  const [maxIterations, setMaxIterations] = useState(100);
  const [classLabels, setClassLabels] = useState([0, 1, 2, 3]);
  const [weights, setWeights] = useState({});
  const [decisionBoundaries, setDecisionBoundaries] = useState([]);
  const [SSE, setSSE] = useState(0);
  const [MSE, setMSE] = useState(0);
  const [selectedLabel, setSelectedLabel] = useState(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');

    // Clear canvas
    context.clearRect(0, 0, canvas.width, canvas.height);

    // Draw data points
    dataPoints.forEach(({ x, y, label }) => {
      context.beginPath();
      context.arc(x, y, 3, 0, 2 * Math.PI);
      context.fillStyle = getColor(label);
      context.fill();
      context.closePath();
    });

    // Draw decision boundaries
    decisionBoundaries.forEach((boundary) => {
      context.beginPath();
      context.moveTo(boundary.start.x, boundary.start.y);
      context.lineTo(boundary.end.x, boundary.end.y);
      context.strokeStyle = 'black';
      context.lineWidth = 2;
      context.stroke();
      context.closePath();
    });
  }, [dataPoints, decisionBoundaries]);

  const handleCanvasClick = (event) => {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    const newDataPoint = { x, y, label: selectedLabel };
    setDataPoints((prevDataPoints) => [...prevDataPoints, newDataPoint]);
  };

  const getColor = (label) => {
    const colors = ['blue', 'red', 'green', 'orange'];
    return colors[label - 1] || 'black';
  };

  const clear = () => {
    setDataPoints([]);
    setDecisionBoundaries([]);
    setSelectedLabel(1);
    setSSE(0);
    setMSE(0);
  };

  const trainModel = () => {
    const input = dataPoints.map(({ x, y }) => [x, y]);
    const labels = dataPoints.map(({ label }) => label);
  
    const classifiers = perceptronLearningOneVsAll(input, labels, learningRate, maxIterations);
  
    const newDecisionBoundaries = [];
    let sseSum = 0;
  
    console.log('classifiers:', classifiers);
  
    for (let i = 0; i < classLabels.length; i++) {
      const classA = classLabels[i];
  
      const decisionBoundary = {
        start: { x: 0, y: 0 },
        end: { x: 0, y: 0 },
      };
  
      const classifierA = classifiers[classA];
  
      console.log('classifierA:', classifierA);
  
      if (classifierA) {
        const weightsA = classifierA.weights;
        const biasA = classifierA.bias;
  
        // Calculate decision boundary for class A
        decisionBoundary.start.x = 0;
        decisionBoundary.start.y = -(biasA / weightsA[1]);
        decisionBoundary.end.x = 600;
        decisionBoundary.end.y = -(weightsA[0] * decisionBoundary.end.x + biasA) / weightsA[1];
        newDecisionBoundaries.push(decisionBoundary);
  
        // Calculate SSE for class A
        for (let k = 0; k < dataPoints.length; k++) {
          const point = dataPoints[k];
          const prediction = predict([point.x, point.y], weightsA, biasA) === 1 ? classA : -1;
          const squaredError = Math.pow(point.label === prediction ? 0 : 1, 2);
          sseSum += squaredError;
        }
      }
    }
  
    const sse = sseSum / (2 * dataPoints.length);
    const mse = sse / dataPoints.length;
  
    setDecisionBoundaries(newDecisionBoundaries);
    setWeights(classifiers);
    setSSE(sse);
    setMSE(mse);
  };
  
  
  

  return (
    <div className="container">
      <h1 className="title">Binary and Multiclass Classification</h1>
      <div className="canvas-container">
        <canvas
          id="canvas"
          width={600}
          height={400}
          onClick={handleCanvasClick}
          ref={canvasRef}
          style={{
            border: '4px solid #4caf50',
            cursor: 'pointer',
            borderRadius: '12px',
            backgroundColor: '#f9f9f9',
            boxShadow: '0 2px 4px rgba(0, 0, 0, 0.1)',
          }}
        />
        <div className="settings-container">
          <div className="setting-row">
            <label className="setting-label" htmlFor="learningRateInput">
              Learning Rate:
            </label>
            <input
              className="setting-input"
              type="number"
              id="learningRateInput"
              value={learningRate}
              onChange={(e) => setLearningRate(parseFloat(e.target.value))}
            />
          </div>
          <div className="setting-row">
            <label className="setting-label" htmlFor="maxIterationsInput">
              Max Iterations:
            </label>
            <input
              className="setting-input"
              type="number"
              id="maxIterationsInput"
              value={maxIterations}
              onChange={(e) => setMaxIterations(parseInt(e.target.value))}
            />
          </div>
          <div className="setting-row">
            <label className="setting-label" htmlFor="labelSelect">
              Select Label:
            </label>
            <select
              className="setting-select"
              id="labelSelect"
              value={selectedLabel}
              onChange={(e) => setSelectedLabel(parseInt(e.target.value))}
            >
              <option value="">None</option>
              {classLabels.map((label) => (
                <option key={label} value={label}>
                  {label}
                </option>
              ))}
            </select>
          </div>
          <div className="button-container">
            <button className="train-button" onClick={trainModel}>
              Train
            </button>
            <button className="clear-button" onClick={clear}>
              Clear
            </button>
          </div>
        </div>
      </div>
      <div>
        <div>SSE: {SSE}</div>
        <div>MSE: {MSE}</div>
      </div>
    </div>
  );
};

export default PerceptronClassifier;
