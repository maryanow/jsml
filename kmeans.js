function kmeans() {
	var m = mathematics(),
		k,
		n,
		centroids = [],
		clusters = [],
		means = [],
		converged = false,
		dims, 
		currClust,
		i, j;

	my.train = function(data, options) {
		var options = options || {};
		k = options['k'] || 4,
		n = data.length,
		dims = data[0].length;

		data = m.shuffle(data);

		for (i = 0; i < k; i++) {
			centroids.push(data[i]);
			clusters.push([]);
		}

		for (i = 0; i < k; i++) {
			means.push([]);
			for (j = 0; j < dims; j++) {
				means[i].push([]);
			}
		}

		while (!converged) {
			converged = true;

			for (i = 0; i < k; i++) {
				for (j = 0; j < dims; j++) {
					means[i][j] = 0.0;
				}
			}

			// Assign datapoints to nearest cluster
			for (i = 0; i < n; i++) {
				currClust = getCluster(data[i]);

				var minDist = (currClust === -1) ? Number.MAX_VALUE : m.distance(centroids[currClust], data[i]);
				for (j = 0; j < k; j++) {
					if (m.distance(centroids[j], data[i]) < minDist) {
						converged = false;
						minDist = m.distance(centroids[j], data[i]);

						if (currClust !== -1) {
							clusters[currClust].splice(m.arrayIndexOf(clusters[currClust], data[i]), 1);
						}

						clusters[j].push(data[i]);
						currClust = j;
					}
				}
			}

			// Evaluate means for each dimension of each datapoint in each cluster
			for (i = 0; i < n; i++) {
				currClust = getCluster(data[i]);

				for (j = 0; j < dims; j++) {
					means[currClust][j] += data[i][j] / clusters[currClust].length;
				}
			}

			// Set each centroid to be the mean
			for (i = 0; i < k; i++) {
				centroids[i] = means[i];
			}
		}

		return my;
	}

	my.predict = function(point) {
		var prediction = -1;

		if (converged) {
			if (point.length === dims) {
				var minDist = Number.MAX_VALUE;
				for (i = 0; i < k; i++) {
					if (m.distance(centroids[i], point) < minDist) {
						prediction = i;
						minDist = m.distance(centroids[i], point);
					}
				}
			}
			else {
				throw new Error("Training points' dimensions does not match predicting point's dimensions: " + dims + " vs. " + point.length);
			}
		}
		else {
			throw new Error("KMeans must be trained before predicting.");
		}

		return prediction;
	}

	my.score = function() {
		var distortion = 0;

		for (i = 0; i < k; i++) {
			for (j = 0; j < clusters[i].length; j++) {
				distortion += Math.pow(m.distance(centroids[i], clusters[i][j]), 2);
			}
		}

		return (distortion / n);
	}

	function getCluster(point) {
		for (var i = 0; i < k; i++) {
			for (var j = 0; j < clusters[i].length; j++) {
				if (m.arrayEquals(clusters[i][j], point)) {
					return i;
				}
			}
		}

		return -1;
	}

	function my() {}
	return my;
}	
