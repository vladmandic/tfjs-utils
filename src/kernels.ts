import * as tf from '@tensorflow/tfjs';
// import '@tensorflow/tfjs-backend-webgl'; included in tfjs
import * as wasm from '@tensorflow/tfjs-backend-wasm';
// import '@tensorflow/tfjs-backend-webgpu'; // old package
import * as webgpu from '../tfjs/tf-backend-webgpu.fesm.js'; // custom built

export async function log(...msg) {
  const str = () => { // helper function: translates json to human readable string
    if (!Array.isArray(msg)) return msg;
    let line = '';
    for (const entry of msg) {
      if (typeof entry === 'object') line += JSON.stringify(entry).replace(/{|}|"|\[|\]/g, '').replace(/,/g, '&nbsp');
      else line += entry + '&nbsp';
    }
    return line;
  };
  const dt = new Date();
  const ts = `${dt.getHours().toString().padStart(2, '0')}:${dt.getMinutes().toString().padStart(2, '0')}:${dt.getSeconds().toString().padStart(2, '0')}.${dt.getMilliseconds().toString().padStart(3, '0')}`;
  // eslint-disable-next-line no-console
  console.log(ts, ...msg);
  const div = document.getElementById('log')
  if (div) div.innerHTML += ts + '&nbsp &nbsp' + str() + '<br>';
}

function enumerateKernels(backends) {
  const kernels = {};
  for (const backend of backends) {
    const kernelList = tf.getKernelsForBackend(backend);
    for (const kernel of kernelList) {
      if (!kernels[kernel.kernelName]) kernels[kernel.kernelName] = {};
      kernels[kernel.kernelName][kernel.backendName] = true;
    }
  }
  return kernels;
}

function generateKernelsHMTL(backends, kernels) {
  const table = document.createElement('table');
  const th = backends.map((backend) => `<th>${backend}</th>`).join('');
  let html = `
    <!-- <caption>List of TF Kernels implemented for each Backend</caption> -->
    <colgroup>
      <col style="background-color: #3f3f3f">
      <col span="${backends.length}" style="background-color: #1f1f1f; width: 4rem">
    </colgroup>
    <thead>
      <tr>
        <th>&nbsp</th>
        ${th}
      </tr>
    </thead>
    <tbody style="font-size: 0.8rem">
  `;
  for (const kernel of Object.keys(kernels)) {
    const implemented = backends.map((backend) => `<td style="text-align: center; background-color: ${kernels[kernel][backend] ? "darkslategrey" : "maroon"}">${kernels[kernel][backend] ? '&#10004' : '-'}</td>`).join('');
    html += `
        <tr>
          <td style="padding-left: 0.5rem">${kernel}</td>
          ${implemented}
        </tr>`;
  }
  html += `
      </tbody>
    </table>`;
  table.innerHTML = html;
  return table;
}

async function main() {
  log('list of tensorflow/js kernels implemented for each backend')
  log(`tfjs: ${tf.version.tfjs} webgl: ${tf.version['tfjs-backend-webgl']} wasm: ${wasm.version_wasm} webgpu: custom`);
  console.log(webgpu);
  const backends = ['cpu', 'wasm', 'webgl', 'webgpu'];
  const kernels = enumerateKernels(backends);
  log('kernels:', Object.keys(kernels).length);
  console.log(kernels);
  const table = generateKernelsHMTL(backends, kernels);
  document.body.appendChild(table);
}

window.onload = main;
