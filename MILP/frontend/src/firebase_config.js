import { initializeApp } from "firebase/app";
import { getFirestore } from "firebase/firestore";


const firebaseConfig = {
    apiKey: "AIzaSyD5KhzNL7kedVGj2IwIWbQNeytvV67r8RI",
    authDomain: "survey-ea041.firebaseapp.com",
    projectId: "survey-ea041",
    storageBucket: "survey-ea041.firebasestorage.app",
    messagingSenderId: "525369034212",
    appId: "1:525369034212:web:fe41a715ee051f2fe773f1",
    measurementId: "G-65XCHXKSJW"
  };

// Initialize Firebase
const app = initializeApp(firebaseConfig);

// Initialize Cloud Firestore and get a reference to the service
export const db = getFirestore(app);