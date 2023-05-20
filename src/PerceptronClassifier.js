import React, { useState, useEffect, useRef } from 'react';
import { predict,OneVsAll } from './perceptron';

const PerceptronClassifier = () => {
  const canvasRef = useRef(null);
  const [dataPoints, setDataPoints] = useState([]);
  const [learningRate, setLearningRate] = useState(0.1);
  const [maxIterations, setMaxIterations] = useState(100);
  const [classLabels, setClassLabels] = useState([0, 1, 2, 3]);
  const [weights, setWeights] = useState({});
  const [separationLines, setseparationLines] = useState([]);
  const [SSE, setSSE] = useState(0);
  const [MSE, setMSE] = useState(0);
  const [selectedLabel, setSelectedLabel] = useState(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');

    context.clearRect(0, 0, canvas.width, canvas.height);

    dataPoints.forEach(({ x, y, label }) => {
      context.beginPath();
      context.arc(x, y, 3, 0, 2 * Math.PI);
      context.fillStyle = getColor(label);
      context.fill();
      context.closePath();
    });

    separationLines.forEach((boundary) => {
      context.beginPath();
      context.moveTo(boundary.start.x, boundary.start.y);
      context.lineTo(boundary.end.x, boundary.end.y);
      context.strokeStyle = 'black';
      context.lineWidth = 2;
      context.stroke();
      context.closePath();
    });
  }, [dataPoints, separationLines]);

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
    setseparationLines([]);
    setSelectedLabel(1);
    setSSE(0);
    setMSE(0);
  };

  const trainModel = () => {
    const input = dataPoints.map(({ x, y }) => [x, y]);
    const labels = dataPoints.map(({ label }) => label);
  
    const classifiers = OneVsAll(input, labels, learningRate, maxIterations);
  
    const newseparationLines = [];
    let sseSum = 0;
    
    for (let i = 0; i < classLabels.length; i++) {
      const classA = classLabels[i];
  
      const decisionBoundary = {
        start: { x: 0, y: 0 },
        end: { x: 0, y: 0 },
      };
  
      const classifierA = classifiers[classA];
  
  
      if (classifierA) {
        const weightsA = classifierA.weights;
        const thresholdA = classifierA.threshold;
  
        decisionBoundary.start.x = 0;
        decisionBoundary.start.y = -(thresholdA / weightsA[1]);
        decisionBoundary.end.x = 600;
        decisionBoundary.end.y = -(weightsA[0] * decisionBoundary.end.x + thresholdA) / weightsA[1];
        newseparationLines.push(decisionBoundary);
  
        for (let k = 0; k < dataPoints.length; k++) {
          const point = dataPoints[k];
          const prediction = predict([point.x, point.y], weightsA, thresholdA) === 1 ? classA : -1;
          const squaredError = Math.pow(point.label - prediction , 2);
          sseSum += squaredError;
        }
      }
    }
  
    const sse = sseSum / (2 * dataPoints.length);
    const mse = sse / dataPoints.length;
  
    setseparationLines(newseparationLines);
    setWeights(classifiers);
    setSSE(sse);
    setMSE(mse);
  };
  
  
  

  return (
    <div className="container">
      <div >
        <canvas
          id="canvas"
          width={700}
          height={550}
          onClick={handleCanvasClick}
          ref={canvasRef}
          style={{
            marginLeft:"100px",
            border: '4px solid #432c91',
            cursor: 'pointer',
            borderRadius: '20px',
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
              min="0.1"
              max="1"
              step="0.1"
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
               min="100"
              max="1000000"
              step="100"
              value={maxIterations}
              onChange={(e) => setMaxIterations(parseInt(e.target.value))}
            />
          </div>
          <div className="setting-row">
            <label className="setting-label" htmlFor="labelSelect" style={{
              marginRight:"25px"
            }}>
              Select Label:    
            </label>
            <select
              className="setting-select"
              id="labelSelect"
              value={selectedLabel}
              onChange={(e) => setSelectedLabel(parseInt(e.target.value))}
              style={{
              paddingRight:"10px",
                paddingLeft: "10px",
            }}
            >
              <option value="">None</option>
              {classLabels.map((label) => (
                <option key={label} value={label} style={{
                    backgroundColor: getColor(label),
                     color: "white"
                     }}>
                  class{label}
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
          <div>
        <div>SSE: {SSE}</div>
        <div>MSE: {MSE}</div>
          </div>

        </div>
      </div>
     
    </div>
  );
};

export default PerceptronClassifier;
