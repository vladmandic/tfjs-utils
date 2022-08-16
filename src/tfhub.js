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
  const json = JSON.parse(stripped);
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
    models.push({
      name: m[0],
      desc: m[4],
      meta: m[7],
      tags: m[14],
    });
  }
  return models;
}

async function get({ model = '' }) {
  log.info('get', model);
  let res;
  let blob;
  let data;
  res = await fetch(`https://tfhub.dev/s/model/1/captain-pool/${encodeURIComponent(model)}?version=`, {
    ...cors,
    headers,
    method: 'GET',
  });
  blob = res && res.ok ? await res.blob() : undefined;
  data = blob ? await parse(blob) : {};
  const modelDetails = {
    details: data[0][1],
    meta: data[0][2],
    description: data[0][4],
    example: data[0][6],
    license: data[0][7],
  };
  res = await fetch(`https://tfhub.dev/s/listModelFormats/captain-pool/${encodeURIComponent(model)}`, {
    ...cors,
    headers,
    method: 'GET',
  });
  blob = res && res.ok ? await res.blob() : undefined;
  data = blob ? await parse(blob) : [];
  const formatList = data[0];
  formatList.shift();
  return { model: modelDetails, formats: formatList[0] };
}

async function main() {
  log.configure({ inspect: { breakLength: 1024, compact: 3, showProxy: true } });
  log.options.timeStamp = false;
  switch (process.argv[2]) {
    case 'list':
      const models = await list({});
      models.sort((a, b) => a.name > b.name ? 1 : -1);
      for (let i = 0; i < models.length; i++) log.blank(i, models[i]);
      break;
    case 'search':
      log.data(await list({ search: process.argv[3] }));
      break;
    case 'get':
      log.data(await get({ model: process.argv[3] }));
      break;
    default:
      log.error('usage: tfhub <list|search|get>');
  }
}

main();
