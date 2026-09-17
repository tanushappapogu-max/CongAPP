export const COCO_LABELS = [
  'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
  'traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat',
  'dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack',
  'umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball',
  'kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket',
  'bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple',
  'sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake',
  'chair','couch','potted plant','bed','dining table','toilet','tv','laptop',
  'mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink',
  'refrigerator','book','clock','vase','scissors','teddy bear','hair drier',
  'toothbrush',
];

const COCO_KNOWN = new Set(COCO_LABELS);

export const TARGET_ALIASES = {
  phone: 'cell phone', iphone: 'cell phone', android: 'cell phone',
  'my phone': 'cell phone', 'cell phone': 'cell phone', mobile: 'cell phone',
  smartphone: 'cell phone', 'phone case': 'cell phone',
  'computer mouse': 'mouse', trackpad: 'mouse', 'my mouse': 'mouse',
  tv: 'tv', television: 'tv', monitor: 'tv', screen: 'tv',
  sofa: 'couch', couch: 'couch', loveseat: 'couch', sectional: 'couch',
  laptop: 'laptop', computer: 'laptop', macbook: 'laptop', notebook: 'laptop',
  labtop: 'laptop',
  remote: 'remote', 'tv remote': 'remote', 'remote control': 'remote',
  ship: 'boat', kayak: 'boat', canoe: 'boat', sailboat: 'boat', yacht: 'boat',
  'jet ski': 'boat',
  cup: 'cup', mug: 'cup', glass: 'cup', 'measuring cup': 'cup',
  'shot glass': 'cup',
  bottle: 'bottle', 'water bottle': 'bottle', thermos: 'bottle',
  'protein shaker': 'bottle', shampoo: 'bottle', conditioner: 'bottle',
  'body wash': 'bottle', lotion: 'bottle', sunscreen: 'bottle',
  'pill bottle': 'bottle',

  // seating
  armchair: 'chair', recliner: 'chair', stool: 'chair', barstool: 'chair',
  'office chair': 'chair', 'patio chair': 'chair', 'adirondack chair': 'chair',

  // tables
  table: 'dining table', 'outdoor table': 'dining table',
  'picnic table': 'dining table',

  // beds
  'twin bed': 'bed', 'full bed': 'bed', 'queen bed': 'bed', 'king bed': 'bed',
  'bunk bed': 'bed', daybed: 'bed', futon: 'bed', crib: 'bed', cot: 'bed',
  mattress: 'bed', 'bed frame': 'bed', 'hospital bed': 'bed',

  // clocks
  'alarm clock': 'clock', 'wall clock': 'clock', 'grandfather clock': 'clock',
  'mantel clock': 'clock',

  // grooming
  'hair dryer': 'hair drier',

  // plants
  plant: 'potted plant', houseplant: 'potted plant', 'flower pot': 'potted plant',
  planter: 'potted plant',

  // kitchen appliances
  fridge: 'refrigerator', freezer: 'refrigerator',
  stove: 'oven', 'toaster oven': 'oven',
  faucet: 'sink',

  // kitchenware
  'kitchen scissors': 'scissors',
  'measuring spoon': 'spoon',
  'mixing bowl': 'bowl',
  'champagne glass': 'wine glass',
  'chef knife': 'knife', 'bread knife': 'knife', 'paring knife': 'knife',

  // food
  bagel: 'donut',

  // office / reading
  textbook: 'book', dictionary: 'book', magazine: 'book', journal: 'book',
  sketchbook: 'book',

  // accessories
  'bow tie': 'tie', 'patio umbrella': 'umbrella',
  purse: 'handbag', 'tote bag': 'handbag', clutch: 'handbag',
  briefcase: 'suitcase', luggage: 'suitcase', 'attach case': 'suitcase',

  // sports balls
  basketball: 'sports ball', football: 'sports ball', 'soccer ball': 'sports ball',
  baseball: 'sports ball', softball: 'sports ball', 'tennis ball': 'sports ball',
  'golf ball': 'sports ball', volleyball: 'sports ball', 'bowling ball': 'sports ball',
  'rugby ball': 'sports ball',

  // sports equipment
  'badminton racket': 'tennis racket', 'cricket bat': 'baseball bat',
  longboard: 'skateboard', ski: 'skis', paddleboard: 'surfboard',

  // bikes
  bike: 'bicycle', 'mountain bike': 'bicycle', 'road bike': 'bicycle',
  'stationary bike': 'bicycle', 'electric bike': 'bicycle',

  // road vehicles
  sedan: 'car', suv: 'car', minivan: 'car', 'station wagon': 'car',
  convertible: 'car', 'sports car': 'car', 'electric car': 'car',
  'hybrid car': 'car', taxi: 'car', uber: 'car',
  van: 'truck', 'pickup truck': 'truck',
  moped: 'motorcycle',
  'school bus': 'bus',
  subway: 'train', tram: 'train',

  // birds
  parrot: 'bird', canary: 'bird', chicken: 'bird', duck: 'bird',
  turkey: 'bird', goose: 'bird', eagle: 'bird', owl: 'bird',
  hawk: 'bird', crow: 'bird', pigeon: 'bird', sparrow: 'bird',

  // other animals
  goat: 'sheep',

  // toys
  'stuffed animal': 'teddy bear',
};

const FUZZY_THRESHOLD = 0.42;

export function normalizeTargetText(text) {
  return (text || '')
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

export function stringSimilarity(a, b) {
  const normA = normalizeTargetText(a);
  const normB = normalizeTargetText(b);
  if (!normA || !normB) return 0;
  if (normA === normB) return 1;

  const overlap = (itemsA, itemsB) => {
    if (!itemsA.length || !itemsB.length) return 0;
    const setA = new Set(itemsA);
    const setB = new Set(itemsB);
    let intersection = 0;
    setA.forEach(item => { if (setB.has(item)) intersection++; });
    return intersection / (setA.size + setB.size - intersection || 1);
  };
  const bigrams = text => {
    const compact = text.replace(/\s+/g, '');
    const grams = [];
    for (let i = 0; i < compact.length - 1; i++) grams.push(compact.slice(i, i + 2));
    return grams;
  };

  const tokenScore = overlap(normA.split(' '), normB.split(' '));
  const gramScore = overlap(bigrams(normA), bigrams(normB));
  const prefixScore = normA.startsWith(normB) || normB.startsWith(normA) ? 0.85 : 0;
  return Math.min(1, Math.max(tokenScore * 0.9 + gramScore * 0.4, prefixScore));
}

export function resolveCocoTarget(tgt) {
  if (!tgt) return null;
  const norm = normalizeTargetText(tgt);
  if (!norm) return null;
  if (COCO_KNOWN.has(norm)) return norm;
  return TARGET_ALIASES[norm] || null;
}

export function findClosestCocoLabel(text) {
  const norm = normalizeTargetText(text);
  if (!norm) return null;
  if (TARGET_ALIASES[norm]) return { label: TARGET_ALIASES[norm], score: 1 };
  if (COCO_KNOWN.has(norm)) return { label: norm, score: 1 };
  let best = { label: null, score: 0 };
  for (const label of COCO_LABELS) {
    const score = stringSimilarity(norm, label);
    if (score > best.score) best = { label, score };
  }
  return best.score >= FUZZY_THRESHOLD ? best : null;
}

