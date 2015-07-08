function linearRegression() {
	var n,
		dims,
		beta_0,
		beta_1,
		datapoints;

	my.train = function(data, options) {
		n = data.length,
		dims = data[0].length,
		datapoints = data;
		
		var sumXY = 0,
			sumX = 0,
			sumY = 0,
			sumXX = 0,
			Sxx, Sxy;

		for (var i = 0; i < n; i++) {
			var x = data[i][0],
				y = data[i][1];

			sumX += x;
			sumY += y;
			sumXX += x * x;
			sumXY += x * y;
		}

		Sxy = sumXY - (sumX * sumY) / n;
		Sxx = sumXX - (sumX * sumX) / n;

		beta_1 = Sxy / Sxx;
		beta_0 = (sumY / n) - (beta_1 * (sumX / n));

		return my;
	}

	my.predict = function(x) {
		return (beta_0 + (beta_1 * x));
	}

	my.r2 = function() {
		var SSR = 0,
			SST = 0,
			yBar = 0;

		for (var i = 0; i < n; i++) {
			yBar += datapoints[i][1] / n;
		}

		for (var i = 0; i < n; i++) {
			SST += Math.pow(datapoints[i][1] - yBar, 2);
		}

		for (var i = 0; i < n; i++) {
			SSR += Math.pow(my.predict(datapoints[i][0]) - yBar, 2);
		}

		return (SSR / SST);
	}

	my.intercept = function() {
		return beta_0;
	}

	my.slope = function() {
		return beta_1;
	}

	function my(){}
	return my;
}