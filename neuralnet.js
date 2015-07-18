function neuralNetwork() {
	var math = mathematics(),
		data,
		output,
		n,
		dims,
		mode,
		classes,
		hiddenNodes,
		W,
		U;

	model.train = function(input, out, options) {
		var options = options || {};

		data = input,
		output = out,
		n = data.length,
		dims = data[0].length,
		mode = options["mode"] || "c",
		classes = output[0].length,
		hiddenNodes = options["hiddenNodes"] || 8;

		var	maxIter = options["maxIter"] || 500,
			standardize = options["standardize"] || false,
			batch = options["batch"] || 1,
			momentum = options["momentum"] || true,
			alpha = options["alpha"] || .9,
			stepSize = options["stepSize"] || .3,
			maxEpochs = options["maxEpochs"] || 20,
			a = math.zeroVector(hiddenNodes),
			h = math.zeroVector(hiddenNodes+1),
			b = math.zeroVector(classes),
			y = math.zeroVector(classes),
			deltaB = math.zeroVector(classes),
			deltaA = math.zeroVector(hiddenNodes),
			gradU = math.zeroMatrix(classes, hiddenNodes+1),
			gradW = math.zeroMatrix(hiddenNodes, dims+1),
			moGradU = math.zeroMatrix(classes, hiddenNodes+1),
			moGradW = math.zeroMatrix(hiddenNodes, dims+1),
			noBiasU = math.zeroMatrix(classes, hiddenNodes),
			noBiasH = math.zeroVector(hiddenNodes);

		W = math.zeroMatrix(hiddenNodes, dims+1);
		U = math.zeroMatrix(classes, hiddenNodes+1);

		h[0] = 1;

		if (standardize) {
			data = standardizeData(data);
		}

		// Add bias to every input
		for (var i = 0; i < n; i++) {
			data[i].unshift(1);
		}

		if (batch === 0) {
			batch = n;
		}

		// Randomly initialize U and W
		var maxW = (1 / Math.sqrt(dims + 1));
		for (var i = 0; i < hiddenNodes; i++) {
			for (var j = 0; j < dims + 1; j++) {
				W[i][j] = ((Math.random() * 2) * maxW) - maxW;
			}
		}
		var maxU = (1 / Math.sqrt(hiddenNodes + 1));
		for (var j = 0; j < classes; j++) {
			for (var k = 0; k < hiddenNodes + 1; k++) {
				U[j][k] = ((Math.random() * 2) * maxU) - maxU;
			}
		}

		var epochs = 0,
			count = 0,
			input, targets;

		while (epochs < maxEpochs) {
			for (var i = 0; i < batch; i++) {
				input = data[count];
				targets = output[count];

				for (var l = 0; l < hiddenNodes; l++) {
					a[l] = math.dotProduct(W[l], input);
					h[l+1] = math.sigmoid(a[l]);
				}

				for (var c = 0; c < classes; c++) {
					b[c] = math.dotProduct(U[c], h);
				}

				for (var c = 0; c < classes; c++) {
					y[c] = outputActivation(b, c, mode);
				}

				deltaB = math.subtractVectors(targets, y);
				gradU = math.addMatrices(gradU, math.outerProduct(deltaB, math.vectorMultiplyScalar(h, -1)));

				for (var j = 0; j < U.length; j++) {
					for (var k = 1; k < U[0].length; k++) {
						noBiasU[j][k-1] = U[j][k];
					}
				}

				for (var j = 1; j < h.length; j++) {
					noBiasH[j-1] = h[j];
				}

				deltaA = math.vectorMultiplyElementwise(math.outerProduct(noBiasU, deltaB), math.vectorMultiplyElementwise(noBiasH, math.subtractVectors(math.oneVector(hiddenNodes), noBiasH)));
				gradW = math.addMatrices(gradW, math.outerProduct(deltaA, math.vectorMultiplyScalar(input, -1)));

				count = (count+1) % n;

				if (count === 0) {
					epochs++;
				}

				// Average gradients over batch and multiply by step size
				gradU = math.matrixMultiplyScalar(gradU, stepSize / batch);
				gradW = math.matrixMultiplyScalar(gradW, stepSize / batch);
			
				if (momentum) {
					moGradW = math.addMatrices(math.matrixMultiplyScalar(moGradW, alpha), gradW);
					moGradU = math.addMatrices(math.matrixMultiplyScalar(moGradU, alpha), gradU);

					W = math.subtractMatrices(W, moGradW);
					U = math.subtractMatrices(U, moGradU);
				}

				// Reset gradients 
				gradU = math.matrixMultiplyScalar(gradU, 0);
				gradW = math.matrixMultiplyScalar(gradW, 0);
			}
		}

		return model;
	}

	model.score = function() {
		var score = 0,
			a = math.zeroVector(hiddenNodes),
			h = math.zeroVector(hiddenNodes+1),
			b = math.zeroVector(classes),
			y = math.zeroVector(classes);

		for (var i = 0; i < n; i++) {
			var input = data[i],
				targets = output[i];
			
			for (var l = 0; l < hiddenNodes; l++) {
				a[l] = math.dotProduct(W[l], input);
				h[l+1] = math.sigmoid(a[l]);
			}

			for (var c = 0; c < classes; c++) {
				b[c] = math.dotProduct(U[c], h);
			}

			for (var c = 0; c < classes; c++) {
				y[c] = outputActivation(b, c);
			}

			score += computeLoss(input, targets, y);
		}

		return score;
	}

	model.predict = function(point) {
		var a = math.zeroVector(hiddenNodes),
			h = math.zeroVector(hiddenNodes+1),
			b = math.zeroVector(classes),
			y = math.zeroVector(classes);

		point.unshift(1);
		h[0] = 1;

		for (var l = 0; l < hiddenNodes; l++) {
			a[l] = math.dotProduct(W[l], point);
			h[l+1] = math.sigmoid(a[l]);
		}

		for (var c = 0; c < classes; c++) {
			b[c] = math.dotProduct(U[c], h);
		}

		for (var c = 0; c < classes; c++) {
			y[c] = outputActivation(b, c);
		}

		return y;
	}

	function outputActivation(b, c) {
		if (mode === "c") {		// Softmax for classification
			var denom = 0;
			for (var k = 0; k < classes; k++) {
				denom += Math.exp(b[k]);
			}

			return (Math.exp(b[c]) / denom);
		}
		else if (mode === "r") {		// Identity for regression
			return b[c];
		}
		else if (mode === "l") {		// Sigmoid for logistic
			return sigmoid(b[c]);
		}
		else {
			throw new Error("Mode not recognized.");
		}
	}

	function computeLoss(point, targets, prediction) {
		var sum = 0;

		if (mode === "c") {
			for (var c = 0; c < classes; c++) {
				sum -= targets[c] * Math.log(prediction[c]);
			}
		}
		else if (mode === "r") {
			for (var c = 0; c < classes; c++) {
				sum += Math.pow(targets[c] - prediction[c], 2);
			}
		}
		else {
			for (var c = 0; c < classes; c++) {
				sum -= ((targets[c] * Math.log(prediction[c])) + ((1 - targets[c]) * Math.log(1 - prediction[c])));
			}
		}

		return sum;		
	}

	function standardizeData(data) {
		var mean = math.zeroVector(dims),
			variance = math.zeroVector(dims),
			result = math.zeroMatrix(n, dims);

		for (var d = 0; d < dims; d++) {
			for (var i in data) {
				mean[d] += data[i][d] / n;
			}
		}

		for (var d = 0; d < dims; d++) {
			for (var i in data) {
				variance[d] += Math.pow(data[i][d] - mean[d], 2) / (n-1);
			}
		}

		for (var i in data) {
			for (var d = 0; d < dims; d++) {
				result[i][d] = (data[i][d] - mean[d]) / variance[d];
			}
		}

		return result;
	}

	function model() {}
	return model;
}