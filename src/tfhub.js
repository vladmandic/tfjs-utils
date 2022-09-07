const fs = require('fs');
const process = require('process');
const log = require('@vladmandic/pilogger');

const headers = {
  accept: 'application/json, text/plain, */*',
  'accept-language': 'en-US,en;q=0.9,hr;q=0.8,sl;q=0.7',
  'cache-control': 'no-cache',
  pragma: 'no-cache',
  'sec-ch-ua': '"Chromium";v="104", " Not A;Brand";v="99", "Microsoft Edge";v="104"',
  'sec-ch-ua-mobile': '?0',
  'sec-ch-ua-platform': '"Windows"',
  'sec-fetch-dest': 'empty',
  'sec-fetch-mode': 'cors',
  'sec-fetch-site': 'same-origin',
};

const cors = {
  referrer: 'https://tfhub.dev/s?subtype=module,placeholder',
  referrerPolicy: 'strict-origin-when-cross-origin',
  mode: 'cors',
  credentials: 'omit',
};

const parse = async (blob) => {
  const text = await blob.text();
  const stripped = text.substring(4);
  let json = {};
  try {
    json = JSON.parse(stripped);
  } catch { /**/ }
  return json;
};

async function list({ search = '', top = 10000 }) {
  log.info('search', search, top);
  const res = await fetch('https://tfhub.dev/s/list', {
    ...cors,
    headers: { ...headers, 'content-type': 'application/json' },
    method: 'POST',
    body: `[["${search}"],[[],[],[],[],[]],[["",${top},true]]]`,
  });
  const blob = res && res.ok ? await res.blob() : undefined;
  const json = blob ? await parse(blob) : {};
  const models = [];
  for (const m of json[0][2][0]) {
    const model = {
      name: m[0],
      pub: m[1],
      desc: m[4].replace('\n', ' '),
      dl: m[7][0],
      meta: m[7].filter((rec) => rec)?.[2],
      tags: m[14],
    };
    model.url = `https://tfhub.dev/${model.pub}/${model.name}`;
    models.push(model);
  }
  return models;
}

async function get({ model = '' }) {
  log.info('get', model);
  let res;
  let blob;
  let data;
  res = await fetch(`https://tfhub.dev/s/model/1/${model}?version=`, {
    ...cors,
    headers,
    method: 'GET',
  });
  blob = res && res.ok ? await res.blob() : undefined;
  data = blob ? await parse(blob) : {};
  let modelDetails = {};
  if (data[0]) {
    modelDetails = {
      details: data[0][1],
      meta: data[0][2],
      description: data[0][4],
      example: data[0][6],
      license: data[0][7]?.[0],
    };
  }
  res = await fetch(`https://tfhub.dev/s/listModelFormats/${model}`, {
    ...cors,
    headers,
    method: 'GET',
  });
  blob = res && res.ok ? await res.blob() : undefined;
  data = blob ? await parse(blob) : [];
  let formatList = {};
  if (data[0]) {
    formatList = data[0];
    formatList.shift();
  }
  return [{ model: modelDetails, formats: formatList[0] }];
}

async function compare({ newList, file }) {
  if (!fs.existsSync(file)) {
    log.error('file does not exist', file);
    return [];
  }
  const bytes = fs.readFileSync(file);
  const json = JSON.parse(bytes);
  log.info({ current: newList.length, previous: json.length });
  const modelsOld = json.map((m) => m.name);
  const modelsNew = newList.map((m) => m.name);
  const lst = [];
  for (const model of modelsNew) {
    if (!modelsOld.includes(model)) {
      lst.push(newList.find((m) => m.name === model));
    }
  }
  return lst;
}

async function main() {
  log.configure({ inspect: { breakLength: 1024, compact: 3, showProxy: true } });
  log.options.timeStamp = false;
  let data = [];
  switch (process.argv[2]) {
    case 'list':
      data = await list({});
      break;
    case 'write':
      data = await list({});
      for (let i = 0; i < data.length; i++) {
        const details = await get({ model: `${data[i].pub}/${data[i].name}` });
        data[i].size = details[0].model?.meta?.[7] || 0;
        data[i].details = details[0].model;
        data[i].formats = details[0].formats;
      }
      fs.writeFileSync(process.argv[3], JSON.stringify(data));
      break;
    case 'read':
      const bytes = fs.readFileSync(process.argv[3]);
      const json = JSON.parse(bytes);
      data = json.sort((a, b) => a.size - b.size).map((m) => ({ name: `${m.pub}/${m.name}`, size: m.size, tag: m.tags }));
      break;
    case 'search':
      data = await list({ search: process.argv[3] });
      break;
    case 'find':
      data = await list({});
      data = data.filter((m) => JSON.stringify(m).toLowerCase().includes(process.argv[3]));
      break;
    case 'get':
      data = await get({ model: process.argv[3] });
      break;
    case 'compare':
      const newList = await list({});
      data = await compare({ newList, file: process.argv[3] });
      break;
    default:
      log.error('usage: tfhub <list|search|find|get|details|compare>');
  }
  if (data && data.length > 0) {
    if (data[0].name) data.sort((a, b) => a.name > b.name ? 1 : -1);
    if (data[0].size) data.sort((a, b) => a.size > b.size ? 1 : -1);
    for (let i = 0; i < data.length; i++) log.blank(data[i]);
    log.info({ records: data.length });
  }
}

main();
