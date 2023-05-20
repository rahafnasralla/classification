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
            weights[j] += learningRate * (actual - prediction) * data[i][j];
          }
          bias += learningRate * (actual - prediction);
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
  export function predict(input, weights, bias) {
    const activation = input.reduce((sum, value, index) => sum + value * weights[index], 0) + bias;
    return activation >= 0 ? 1 : -1;
  }
  
  
  export function perceptronLearningOneVsAll(data, labels, learningRate, maxIterations) {
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
  
  
  