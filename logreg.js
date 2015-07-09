function logisticRegression() {
	var n,
		dims,
		beta,
		cl,
		math = mathematics();

	my.train = function(data, output, options) {
		if (data.length !== output.length) {
			throw new Error("Number of datapoints and number of outputs must match.");
		}

		options = options || {};

		var classes = [];
		for (var i in output) {
			if (classes.indexOf(output[i][0]) === -1) {
				classes.push(output[i][0]);
			}
		}

		cl = classes.length;

		if (cl === 1) {
			throw new Error("Output must consist of more than one class.");
		}
		else if (cl === 2) {
			binaryLogReg(data, output, options);
		}
		else {
			multinomialLogReg(data, output, options);
		}

		return my;
	}

	my.predict = function(point) {
		point.unshift(1);

		if (cl > 2) {
			var current = -1,
				max = 0;

			for (var c = 0; c < cl; c++) {
				if (math.sigmoid(math.dotProduct(beta[c], point)) > max) {
					current = c;
					max = math.sigmoid(math.dotProduct(beta[c], point));
				}
			}

			return [current, max];
		}
		else {
			return math.sigmoid(math.dotProduct(beta, point));
		}
	}

	my.beta = function() {
		return beta;
	}

	function binaryLogReg(data, output, options) {
		var options = options || {},
			maxIter = options["maxIter"] || 500,
			stepSize = options["stepSize"] || .25,
			sum;

			n = Math.min(data.length, 500),
			dims = data[0].length,
			beta = math.zeroVector(dims + 1);

		// Add bias to each datapoint
		for (var i = 0; i < n; i++) {
			data[i].unshift(1);
		}

		for (var i = 0; i < maxIter; i++) {
			sum = math.zeroVector(dims + 1);

			// Calculate gradient
			for (var j = 0; j < n; j++) {
				var diff = output[j] - math.sigmoid(math.dotProduct(data[j], beta));
				var mult = math.vectorMultiplyScalar(data[j], diff);
				sum = math.addVectors(sum, mult);
			}

			sum = math.vectorMultiplyScalar(sum, stepSize/n);
			beta = math.addVectors(beta, sum);
		}
	}

	function multinomialLogReg(data, output, options) {
		n = Math.min(data.length, 500),
		dims = data[0].length,
		beta = math.zeroMatrix(cl, dims + 1);

		var options = options || {},
			maxIter = options["maxIter"] || 500,
			stepSize = options["stepSize"] || .25,
			sum,
			denom,
			probs = math.zeroMatrix(n, cl);

		// Add bias to each datapoint
		for (var i = 0; i < n; i++) {
			data[i].unshift(1);
		}

		for (var i = 0; i < maxIter; i++) {
			for (var t = 0; t < n; t++) {
				denom = 0;
				for (var c = 0; c < cl; c++) {
					denom += Math.exp(math.dotProduct(beta[c], data[t]));
				}

				for (var c = 0; c < cl; c++) {
					probs[t][c] = Math.exp(math.dotProduct(beta[c], data[t])) / denom;
				}
			}

			for (var c = 0; c < cl; c++) {
				var gradient = math.zeroVector(dims + 1);

				for (var t = 0; t < n; t++) {
					if (output[t] == c) {
						gradient = math.addVectors(gradient, math.vectorMultiplyScalar(data[t], 1 - probs[t][c]));
					}
					else {
						gradient = math.addVectors(gradient, math.vectorMultiplyScalar(data[t], 0 - probs[t][c]));
					}
				}

				gradient = math.vectorMultiplyScalar(gradient, stepSize/n);
				beta[c] = math.addVectors(beta[c], gradient);
			}
		}
	}

	function my(){}
	return my;
}