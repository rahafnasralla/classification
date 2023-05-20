function perceptron(data, classes, learningRate, maxIterations) {

    function Random() {
        return Math.random() - 0.5;
      }
    
    let weights = [Random(),Random()];
    let threshold = Random();
  
    let iteration = 0;
  
    while (iteration < maxIterations) {
      let errorCount = 0;
  
      for (let i = 0; i < data.length; i++) {
        const actual = activate(data[i], weights, threshold);
        const desired = classes[i];
  
        if (actual !== desired) {
          for (let j = 0; j < weights.length; j++) {
            weights[j] += learningRate * (desired - actual) * data[i][j];
          }
          threshold += learningRate * (desired - actual);
          errorCount++;
        }
      }
  
      if (errorCount === 0) {
        break;
      }
  
      iteration++;
    }
  
    return { weights, threshold };
  }
  
  export function activate(input, weights, threshold) {
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
        const { weights, threshold } = perceptron(data, binaryClasses, learningRate, maxIterations);
        classifiers.push({ class_: uniqueclasses[0], weights, threshold });
      } else {

    for (let i = 0; i < uniqueclasses.length; i++) {
      const currentclass_ = uniqueclasses[i];
      const binaryclasses = classes.map((class_) => (class_ === currentclass_ ? 1 : -1));
      const { weights, threshold } = perceptron(data, binaryclasses, learningRate, maxIterations);
      classifiers.push({ class_: currentclass_, weights, threshold });
    }
}
    return classifiers;
  }
  
  
  
  