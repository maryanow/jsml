function mathematics() {
	math.shape = function(container) {
		return [container.length, (container[0].length || 1)];
	}

	math.transpose = function(matrix) {
		var result = math.zeroVector(matrix[0].length);
		for (var i = 0; i < matrix[0].length; i++) {
			result[i] = math.zeroVector(matrix.length);
			for (var j = 0; j < matrix.length; j++) {
				result[i][j] = matrix[j][i];
			}
		}

		return result;
	}

	math.identity = function(size) {
		var result = math.zeroMatrix(size, size);
		for (var i = 0; i < size; i++) {
			for (var j = 0; j < size; j++) {
				if (i === j) {
					result[i][j] = 1;
				}
			}
		}

		return result;
	}

	math.inverse = function(matrix) {
		if (matrix.length !== matrix[0].length) {
			throw new Error("Matrix must be square.");
		}

		var inv = math.identity(matrix.length),
			copy = math.copyMatrix(matrix);

		for (var i = 0; i < matrix.length; i++) {
			var cell = copy[i][i];
			if (cell === 0) {
				for (j = i+1; j < matrix.length; j++) {
					if (copy[j][i] !== 0) {
						for (var k = 0; k < matrix.length; k++) {
							cell = copy[i][k];
							copy[i][k] = copy[j][k];
							copy[j][k] = cell;
							cell = inv[i][k];
							inv[i][k] = inv[j][k];
							inv[j][k] = cell;
						}

						break;
					}
				}

				if (copy[i][i] === 0) {
					throw new Error("Matrix is not invertable.");
				}
			}

			for (var j = 0; j < matrix.length; j++) {
				copy[i][j] = copy[i][j] / cell;
				inv[i][j] = inv[i][j] / cell;
			}

			for (var j = 0; j < matrix.length; j++) {
				if (j === i) {
					continue;
				}

				cell = copy[j][i];

				for (var k = 0; k < matrix.length; k++) {
					copy[j][k] -= cell * copy[i][k];
					inv[j][k] -= cell * inv[i][k];
				}
			}
		}

		return inv;
	}

	math.copyArray = function(array) {
		var result = [];
		for (var i = 0; i < array.length; i++) {
			result.push(array[i]);
		}

		return result;
	}

	math.copyMatrix = function(matrix) {
		var result = [];
		for (var i = 0; i < matrix.length; i++) {
			result.push(math.copyArray(matrix[i]));
		}

		return result;
	}

	math.arrayEquals = function(v1, v2) {
		if (v1.length !== v2.length) {
			return false;
		}

		for (var i = 0; i < v1.length; i++) {
			if (v1[i] !== v2[i]) {
				return false;
			}
		}

		return true;
	}

	math.matrixEquals = function(m1, m2) {
		if (m1.length !== m2.length || m1[0].length !== m2[0].length) {
			return false;
		}

		for (var i = 0; i < m1.length; i++) {
			if (!math.arrayEquals(m1[i], m2[i])) {
				return false;
			}
		}

		return true;
	}

	math.arrayIndexOf = function(container, array) {
		for (var i = 0; i < container.length; i++) {
			if (math.arrayEquals(container[i], array)) {
				return i;
			}
		}

		return -1;
	}

	math.addVectors = function(v1, v2) {
		if (v1.length === v2.length) {
			var result = [];
			for (var i = 0; i < v1.length; i++) {
				result.push(parseFloat(v1[i]) + parseFloat(v2[i]));
			}

			return result;
		}
		else {
			throw new Error("Vectors must have the same length.");
		}
	}

	math.subtractVectors = function(v1, v2) {
		if (v1.length === v2.length) {
			var result = [];
			for (var i = 0; i < v1.length; i++) {
				result.push(v1[i] - v2[i]);
			}

			return result;
		}
		else {
			throw new Error("Vectors must have the same length.");
		}
	}

	math.addMatrices = function(m1, m2) {
		if (math.arrayEquals(math.shape(m1), math.shape(m2))) {
			var result = math.zeroMatrix(m1.length, m1[0].length);

			for (var i = 0; i < m1.length; i++) {
				for (var j = 0; j < m1[0].length; j++) {
					result[i][j] = m1[i][j] + m2[i][j];
				}
			}

			return result;
		}
		else {
			throw new Error("Matrices must have the same number rows and columns.");
		}
	}

	math.subtractMatrices = function(m1, m2) {
		if (math.arrayEquals(math.shape(m1), math.shape(m2))) {
			var result = math.zeroMatrix(m1.length, m1[0].length);
			for (var i = 0; i < m1.length; i++) {
				for (var j = 0; j < m1[0].length; j++) {
					result[i][j] = m1[i][j] - m2[i][j];
				}
			}

			return result;
		}
		else {
			throw new Error("Matrices must have the same number of rows and columns.");
		}
	}

	math.vectorMultiplyScalar = function(vector, scalar) {
		var result = [];
		for (var i = 0; i < vector.length; i++) {
			result.push(vector[i] * scalar);
		}

		return result;
	}

	math.vectorMultiplyElementwise = function(v1, v2) {
		if (v1.length === v2.length) {
			var result = [];
			for (var i = 0; i < v1.length; i++) {
				result.push(v1[i] * v2[i]);
			}

			return result;
		}
		else {
			throw new Error("Vecotrs must have the same lengths.");
		}
	}

	math.matrixMultiplyScalar = function(matrix, scalar) {
		var result = [];
		for (var i = 0; i < matrix.length; i++) {
			result.push(math.vectorMultiplyScalar(matrix[i], scalar));
		}

		return result;
	}

	math.dotProduct = function(v1, v2) {
		if (v1.length === v2.length) {
			var result = 0;
			for (var i = 0; i < v1.length; i++) {
				result += parseFloat(v1[i] * v2[i]);
			}

			return result;
		}
		else {
			throw new Error("Vectors must have the same length.");
		}
	}

	math.outerProduct = function (matrix, vector) {
		var result,
			sum;

		if (matrix[0].length) {		// Matrix
			result = math.zeroVector(matrix[0].length);

			for (var i = 0; i < matrix[0].length; i++) {
				sum = 0;
				for (var j = 0; j < matrix.length; j++) {
					sum += matrix[j][i] * vector[j];
				}

				result[i] = sum;
			}
		}
		else {						// Vector
			result = math.zeroMatrix(matrix.length, vector.length);

			for (var i = 0; i < matrix.length; i++) {
				for (var j = 0; j < vector.length; j++) {
					result[i][j] = matrix[i] * vector[j];
				}
			}
		}
		
		return result;
	}

	math.zeroVector = function(size) {
		var result = [];
		for (var i = 0; i < size; i++) {
			result.push(0);
		}

		return result;
	}

	math.oneVector = function(size) {
		var result = [];
		for (var i = 0; i < size; i++) {
			result.push(1);
		}

		return result;
	}

	math.zeroMatrix = function(row, col) {
		var result = [];
		for (var i = 0; i < row; i++) {
			result.push(math.zeroVector(col));
		}

		return result;
	}

	math.oneMatrix = function(row, col) {
		var result = [];
		for (var i = 0; i < row; i++) {
			result.push(math.oneVector(col));
		}

		return result;
	}

	math.sigmoid = function(x) {
		return (1 / (1 + Math.exp(-x)));
	}

	math.distance = function(p1, p2) {
		var sum = 0;

		if (p1.length === p2.length) {
			for (var i = 0; i < p1.length; i++) {
				sum += Math.pow(p1[i] - p2[i], 2);
			}

			return Math.sqrt(sum);
		}
	}

	math.shuffle = function(collection) {
		var result = math.copyArray(collection),
			length = collection.length,
			temp, i;

		while (length > 0) {
			i = Math.floor(Math.random() * length);
			length--;

			temp = result[length];
			result[length] = result[i];
			result[i] = temp;
		}

		return result;
	}

	math.determinant = function(matrix) {
		var sum = 0;

		if (matrix.length === 1) {
			return matrix[0][0];
		}

		for (var i = 0; i < matrix.length; i++) {
			var reduced = math.zeroMatrix(matrix.length-1, matrix.length-1);

			for (var j = 1; j < matrix.length; j++) {
				for (var k = 0; k < matrix.length; k++) {
					if (k < i) {
						reduced[j-1][k] = matrix[j][k];
					}
					else if (k > i) {
						reduced[j-1][k-1] = matrix[j][k]
					}
				}
			}

			sum += (i % 2 === 0 ? 1 : -1) * matrix[0][i] * math.determinant(reduced);
		}

		return sum;
	}

	function math(){}
	return math;
}