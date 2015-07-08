function logisticRegression() {
	var n,
		dims,
		beta,
		math = mathematics();

	my.train = function(data, output, options) {
		if (data.length !== output.length) {
			throw new Error("Number of datapoints and number of outputs must match.");
		}

		var options = options || {},
			maxIter = options["maxIter"] || 500,
			stepSize = options["stepSize"] || .25,
			sum;

			n = Math.max(data.length, 500),
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

			// If the gradient is ever 0, then the algorithm has converged
			if (math.vectorsEqual(sum, math.zeroVector(dims + 1))) {
				break;
			}

			sum = math.vectorMultiplyScalar(sum, 1/n);
			sum = math.vectorMultiplyScalar(sum, stepSize);

			beta = math.addVectors(beta, sum);

			console.log("Iter " + i);
		}

		return my;
	}

	my.predict = function(point) {
		point.unshift(1);
		return math.sigmoid(math.dotProduct(beta, point));
	}

	my.score = function() {

	}

	my.beta = function() {
		return beta;
	}

	function my(){}
	return my;
}