// ================================================================
//  FIREBASE CONFIGURATION
//  ================================================================
//  STEP 1 — Go to https://console.firebase.google.com
//  STEP 2 — Select your project → Project Settings → Your apps
//  STEP 3 — Copy the firebaseConfig object and paste it below
//  ================================================================

const firebaseConfig = {
  apiKey:            "YOUR_API_KEY",
  authDomain:        "YOUR_PROJECT_ID.firebaseapp.com",
  projectId:         "YOUR_PROJECT_ID",
  storageBucket:     "YOUR_PROJECT_ID.appspot.com",
  messagingSenderId: "YOUR_MESSAGING_SENDER_ID",
  appId:             "YOUR_APP_ID"
};

// Initialize Firebase app
firebase.initializeApp(firebaseConfig);

// Expose Firestore as a global `db` used by every page script
const db = firebase.firestore();
