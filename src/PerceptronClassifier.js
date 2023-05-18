import React, { useState, useEffect, useRef } from 'react';


const  PerceptronClassifier = () => {
  const canvasRef = useRef(null);
  const [dataPoints, setDataPoints] = useState([]);
  const [learningRate, setLearningRate] = useState(0.1);
  const [maxIterations, setMaxIterations] = useState(100);
  const [classLabels, setClassLabels] = useState([1,2,3,4]);
  const [weights, setWeights] = useState([]);
  const [decisionBoundaries, setDecisionBoundaries] = useState([]);
  const [SSE, setSSE] = useState(0);
  const [MSE, setMSE] = useState(0);

  const [selectedLabel, setSelectedLabel] = useState(1);


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

  const sigmoid = (z) => 1 / (1 + Math.exp(-z));
  const trainModel = () => {
    const uniqueLabels = [...new Set(dataPoints.map(({ label }) => label))];
    setClassLabels(uniqueLabels);

    const n = dataPoints.length;
    const m = uniqueLabels.length;
    const weights = [];
    let sse =0;

    for (let i = 0; i <m; i++) {
      const label = uniqueLabels[i];
      const labelData = dataPoints.map(({ x, y, label: pointLabel }) => [1, x, y, pointLabel === label ? 1 : 0]);
      const labelWeights = [0, 0, 0];
      const labelX = labelData.map((row) => row.slice(0, 3));
      const labelY = labelData.map((row) => row[3]);
      for (let iteration = 0; iteration < maxIterations; iteration++) {
        for (let j = 0; j < n; j++) {
          const predicted = sigmoid(labelWeights.reduce((acc, val, k) => acc + val * labelX[j][k], 0));
          const error = labelY[j] - predicted;
    
          labelWeights[0] += learningRate * error;
          labelWeights[1] += learningRate * error * labelX[j][1];
          labelWeights[2] += learningRate * error * labelX[j][2];
          sse += error ^ 2;
        }
      }
    
      const decisionBoundary = {
        start: { x: 0, y: -(labelWeights[0] + labelWeights[1] * 0) / labelWeights[2] },
        end: { x: canvasRef.current.width, y: -(labelWeights[0] + labelWeights[1] * canvasRef.current.width) / labelWeights[2] },
      };
    
      weights.push(labelWeights);
      setDecisionBoundaries((prevDecisionBoundaries) => [...prevDecisionBoundaries, decisionBoundary]);
    }

    const mse = sse / (n * m);
    setSSE(sse)
    setMSE(mse)
    
    setWeights(weights);
  };
  

  const clear = () => {
    setDataPoints([]);
    setDecisionBoundaries([]);
    setSelectedLabel(1);
    setSSE(0)
    setMSE(0)
  };

  return (
  <div className="container">
  <h1 className="title">Binary and Multiclass Classification</h1>
  <div className='canvas-container'>
  <canvas id="canvas" 
  width={600}
  height={400}
  onClick={handleCanvasClick}
  ref={canvasRef}
  style={{ border: '4px solid #4caf50', cursor: 'pointer', borderRadius: '12px'
 , backgroundColor: '#f9f9f9', boxShadow: '0 2px 4px rgba(0, 0, 0, 0.1)' }} />
<div className="settings-container">
          <div className="setting-row">
            <label className="setting-label" htmlFor="learningRateInput">Learning Rate:</label>
            <input
              className="setting-input"
              type="number"
              id="learningRateInput"
              value={learningRate}
              onChange={(e) => setLearningRate(parseFloat(e.target.value))}
            />
          </div>
          <div className="setting-row">
            <label className="setting-label" htmlFor="maxIterationsInput">Max Iterations:</label>
            <input
              className="setting-input"
              type="number"
              id="maxIterationsInput"
              value={maxIterations}
              onChange={(e) => setMaxIterations(parseInt(e.target.value))}
            />
          </div>
          <div className="setting-row">
            <label className="setting-label" htmlFor="labelSelect">Select Label:</label>
            <select
              className="setting-select"
              id="labelSelect"
              value={selectedLabel}
              onChange={(e) => setSelectedLabel(e.target.value)}
            >
            
              <option value="">None</option>
              {classLabels.map((label) => (
                <option key={label} value={label}>{label}</option>
              ))}
            </select>
          </div>
<div className='button-container'>
<button className='train-button' onClick={trainModel}>Train</button>
<button className='clear-button' onClick={clear}>Clear</button>
</div>
</div>
  </div>

  <div>
<div>
        SSE: {SSE}
      </div>
      <div>
        MSE: {MSE}
      </div>
</div>

  </div>
  );
  };
  
  export default PerceptronClassifier;    