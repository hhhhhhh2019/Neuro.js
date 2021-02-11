'use strict'
function activate(x) {
return 1 / (1 + Math.exp(-x));
}

function measure(x, y, e) {
	return e * y * (1 - y) * x;
}

/*function activate(x) {
	if (x > 1) return 1 + 0.01 * (x - 1)
	if (x < 0) return 0.01 * x;
	if (0 <= x && x <= 1) return x;
}

function measure(x, y, e) {
	let a = 0;
	if (x > 1 || x < 0) {a = 0.01}
	else {a = 1}
	return x * y * e;
}*/

function count_lay(l0, l1, w) {
	for (let i = 0; i < l1.length; i++) {
		l1[i][0] = 0;
		for (let j = 0; j < l0.length; j++) {
			l1[i][0] += l0[j][0] * w[i][j];
		}
		l1[i][0] = activate(l1[i][0]);
	}
}

function find_out_error(l, v) {
	for (let i = 0; i < l.length; i++) {
		l[i][1] = v[i] - l[i][0];
	}
}

function find_error(l0, l1, w) {
	for (let i = 0; i < l0.length; i++) {
		l0[i][1] = 0;
		for (let j = 0; j < l1.length; j++) {
			l0[i][1] += l1[j][1] * w[j][i];
		}
	}
}

function correct_error(l0, l1, w, k) {
	for (let i = 0; i < w.length; i++) {
		for (let j = 0; j < w[i].length; j++) {
			w[i][j] += k * measure(l0[j][0], l1[i][0], l1[i][1]);
		}
	}
}

function _run(lays, weights, data) {<
	for (let i = 0; i < data.length; i++) {
		lays[0][i][0] = data[i];
	}

	for (let i = 0; i < lays.length - 1; i++) {
		count_lay(lays[i], lays[i+1], weights[i]);
	}

	let res = [];
	for (let i of lays[lays.length-1]) {
		res.push(i[0]);
	}

	return res;
}

function get_error(lays) {
	let e = 0;

	for (let i of lays) {
		for (let n of i) {
			e += Math.abs(n[1]);
		}
	}

	return e;
}

function _learn(lays, weights, ld, iters, koof, log) {
	let errors = [];

	for (let k = 0; k < iters; k++) {
		for (let d of ld) {
			_run(lays, weights, d[0]);

			find_out_error(lays[lays.length-1], d[1]);
			for (let i = lays.length-1; i > 1; i--) {
				find_error(lays[i-1], lays[i], weights[i-1]);
			}

			errors.push(get_error(lays));
			let e = 0;
			for (let i of lays[lays.length-1]) {
				e += Math.abs(i[1]);
			}
			if (e < 0.01) continue;

			for (let i = 0; i < lays.length-1; i++) {
				correct_error(lays[i], lays[i+1], weights[i], koof);
			}
		}

		if (log && k % 100 == 0) console.log("осталось: " + iters - k);
	}

	alert("finished");

	return errors;
}


class NeuralNetwork {
	constructor(nc) {
		this.lays = [];
		for (let i of nc) {
			let l = [];
			for (let j = 0; j < i; j++) {
				l.push([0, 0]);
			}
			this.lays.push(l);
		}

		this.weights = [];
		for (let i = 1; i < nc.length; i++) {
			let l = [];
			for (let j = 0; j < nc[i]; j++) {
				let w = [];
				for (let k = 0; k < nc[i-1]; k++) {
					w.push(Math.random()*2-1);
				}
				l.push(w);
			}
			this.weights.push(l);
		}
	}

	run(data) {
		return _run(this.lays, this.weights, data);
	}

	learn(ld, i=10000, k=0.2, l=false) {
		return _learn(this.lays, this.weights, ld, i, k, l);
	}

	evolution(p) {
		for (let i = 0; i < this.weights.length; i++) {
			for (let j = 0; j < this.weights[i].length; j++) {
				for (let k = 0; k < this.weights[i][j].length; k++) {
					this.weights[i][j][k] = p.weights[i][j][k] + (Math.random()-0.5)/3;
				}
			}
		}
	}
}
