/**
 * objectList.js — "Trained Objects" list shown in the UI drawer.
 *
 * Every entry here resolves through detection/coco.js's resolveCocoTarget()
 * to one of the 80 COCO classes the on-device YOLO11n model can detect.
 * Keep it that way — an entry that doesn't resolve will start a scan that
 * can never lock onto anything.
 */

export const KNOWN_OBJECTS = [
  // ── Furniture ─────────────────────────────────────────────────────────────
  'chair', 'armchair', 'recliner', 'sofa', 'couch', 'loveseat', 'sectional',
  'bench', 'stool', 'barstool', 'table', 'dining table',
  'bed', 'twin bed', 'full bed',
  'queen bed', 'king bed', 'bunk bed', 'daybed', 'futon', 'crib', 'cot',
  'mattress', 'bed frame',

  // ── Bedroom items ─────────────────────────────────────────────────────────
  'alarm clock',
  'hair dryer',

  // ── Living room ───────────────────────────────────────────────────────────
  'tv', 'television', 'monitor', 'screen', 'remote control',
  'remote', 'vase', 'clock',
  'wall clock', 'grandfather clock', 'mantel clock', 'plant', 'houseplant',
  'planter',

  // ── Kitchen items ─────────────────────────────────────────────────────────
  'refrigerator', 'fridge', 'freezer', 'stove', 'oven', 'microwave',
  'toaster', 'toaster oven', 'sink', 'faucet',
  'knife', 'chef knife', 'bread knife', 'paring knife',
  'kitchen scissors',
  'measuring cup', 'measuring spoon', 'mixing bowl', 'bowl',
  'cup', 'mug', 'glass', 'wine glass',
  'champagne glass', 'shot glass', 'thermos', 'water bottle',

  // ── Food items ────────────────────────────────────────────────────────────
  'apple', 'banana', 'orange',
  'carrot', 'broccoli',
  'bagel',
  'cake',

  // ── Bathroom items ────────────────────────────────────────────────────────
  'toothbrush',
  'shampoo', 'conditioner', 'body wash', 'lotion', 'sunscreen',
  'toilet',
  'pill bottle',

  // ── Electronics & tech ────────────────────────────────────────────────────
  'laptop', 'computer', 'phone',
  'smartphone', 'iphone', 'phone case',
  'keyboard', 'mouse',

  // ── Office & school supplies ──────────────────────────────────────────────
  'scissors',
  'notebook',
  'office chair',
  'book', 'textbook', 'dictionary', 'magazine', 'journal',

  // ── Clothing & accessories ────────────────────────────────────────────────
  'tie', 'bow tie',
  'purse', 'handbag',
  'tote bag', 'backpack', 'briefcase', 'clutch',
  'suitcase', 'luggage', 'umbrella',

  // ── Sports & fitness ─────────────────────────────────────────────────────
  'basketball', 'football', 'soccer ball', 'baseball', 'softball', 'tennis ball',
  'golf ball', 'volleyball', 'bowling ball', 'rugby ball', 'frisbee',
  'tennis racket', 'badminton racket',
  'baseball bat', 'cricket bat',
  'skateboard', 'longboard',
  'ski', 'snowboard', 'surfboard', 'paddleboard', 'kayak',
  'canoe', 'bike', 'mountain bike', 'road bike',
  'stationary bike',
  'protein shaker',

  // ── Garden & outdoor ─────────────────────────────────────────────────────
  'flower pot',
  'patio chair',
  'adirondack chair', 'outdoor table', 'picnic table',
  'patio umbrella',

  // ── Vehicles & transport ──────────────────────────────────────────────────
  'car', 'sedan', 'suv', 'truck', 'pickup truck', 'van', 'minivan',
  'station wagon', 'convertible', 'sports car', 'electric car', 'hybrid car',
  'motorcycle', 'moped', 'bicycle', 'electric bike',
  'bus', 'school bus', 'subway', 'train', 'tram', 'taxi', 'uber',
  'airplane', 'boat', 'sailboat', 'yacht', 'jet ski',

  // ── Animals ───────────────────────────────────────────────────────────────
  'dog', 'cat', 'bird', 'parrot', 'canary', 'horse', 'cow',
  'sheep', 'goat', 'chicken', 'duck', 'turkey', 'goose',
  'bear',
  'elephant', 'giraffe', 'zebra', 'eagle', 'owl', 'hawk', 'crow', 'pigeon', 'sparrow',

  // ── Art & craft ───────────────────────────────────────────────────────────
  'sketchbook',

  // ── Medical & health ─────────────────────────────────────────────────────
  'hospital bed',

  // ── Toys & games ─────────────────────────────────────────────────────────
  'stuffed animal', 'teddy bear',
  'kite',

  // ── Bags & containers ─────────────────────────────────────────────────────
  'bottle',
  'attaché case',
];

/**
 * Filter the object list to those matching a prefix (case-insensitive).
 * Returns up to `limit` results.
 */
export function suggestObjects(prefix, limit = 8) {
  if (!prefix || prefix.length < 2) return [];
  const q = prefix.toLowerCase().trim();
  return KNOWN_OBJECTS.filter(obj => obj.includes(q)).slice(0, limit);
}

/**
 * Returns true if the given name is in the known objects list.
 */
export function isKnownObject(name) {
  const q = (name || '').toLowerCase().trim();
  return KNOWN_OBJECTS.some(obj => obj === q || obj.includes(q));
}
