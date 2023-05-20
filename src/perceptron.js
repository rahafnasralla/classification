// Perceptron Learning Algorithm
function perceptronLearning(data, classes, learningRate, maxIterations) {
    // Initialize weights and threshold

    function Random() {
        return Math.random() - 0.5;
      }
    
    let weights = [Random(),Random()];
    let threshold = Random();
  
    let iteration = 0;
  
    while (iteration < maxIterations) {
      let errorCount = 0;
  
      // Iterate 
      for (let i = 0; i < data.length; i++) {
        const prediction = predict(data[i], weights, threshold);
        const actual = classes[i];
  
        // Update weights and threshold if prediction is incorrect
        if (prediction !== actual) {
          for (let j = 0; j < weights.length; j++) {
            weights[j] += learningRate * (actual - prediction) * data[i][j];
          }
          threshold += learningRate * (actual - prediction);
          errorCount++;
        }
      }
  
      // If there are no errors, exit the loop
      if (errorCount === 0) {
        break;
      }
  
      iteration++;
    }
  
    return { weights, threshold };
  }
  
  export function predict(input, weights, threshold) {
    let activation = threshold;
    for (let i = 0; i < input.length; i++) {
      activation += input[i] * weights[i];
    }
    return activation >= 0 ? 1 : -1;
  }
  
  export function OneVsAll(data, classes, learningRate, maxIterations) {
    const uniqueclasses = [...new Set(classes)];
    const classifiers = [];
    if ( uniqueclasses.length === 2) {
        const binaryClasses = classes.map((class_) => (class_ === uniqueclasses[0] ? 1 : -1));
        const { weights, threshold } = perceptronLearning(data, binaryClasses, learningRate, maxIterations);
        classifiers.push({ class_: uniqueclasses[0], weights, threshold });
      } else {
    // Train a perceptron for each class
    for (let i = 0; i < uniqueclasses.length; i++) {
      const currentclass_ = uniqueclasses[i];
      const binaryclasses = classes.map((class_) => (class_ === currentclass_ ? 1 : -1));
      const { weights, threshold } = perceptronLearning(data, binaryclasses, learningRate, maxIterations);
      classifiers.push({ class_: currentclass_, weights, threshold });
    }
}
    return classifiers;
  }
  
  
  