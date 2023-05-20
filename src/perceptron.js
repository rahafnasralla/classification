// Perceptron Learning Algorithm
function perceptronLearning(data, classes, learningRate, maxIterations) {
    // Initialize weights and bias

    function Random() {
        return Math.random() - 0.5;
      }
    
    let weights = [0,0];
    let bias = 0;
  
    let iteration = 0;
  
    while (iteration < maxIterations) {
      let errorCount = 0;
  
      // Iterate 
      for (let i = 0; i < data.length; i++) {
        const prediction = predict(data[i], weights, bias);
        const actual = classes[i];
  
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
  
  export function predict(input, weights, bias) {
    const activation = input.reduce((sum, value, index) => sum + value * weights[index], 0) + bias;
    return activation >= 0 ? 1 : -1;
  }
  
  export function OneVsAll(data, classes, learningRate, maxIterations) {
    const uniqueclasses = [...new Set(classes)];
    const classifiers = [];
  
    // Train a perceptron for each class
    for (let i = 0; i < uniqueclasses.length; i++) {
      const currentclass_ = uniqueclasses[i];
      const binaryclasses = classes.map((class_) => (class_ === currentclass_ ? 1 : -1));
      const { weights, bias } = perceptronLearning(data, binaryclasses, learningRate, maxIterations);
      classifiers.push({ class_: currentclass_, weights, bias });
    }
  
    return classifiers;
  }
  
  
  