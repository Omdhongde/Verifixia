/**
 * Firebase Authentication helpers
 * ---------------------------------
 * Wraps Firebase Auth operations so the rest of the app never
 * imports firebase/auth directly.
 *
 * Exports a `firebaseEnabled` flag – when false (no env vars set) every
 * function is a graceful no-op so the app still works locally without Firebase.
 */

import {
  createUserWithEmailAndPassword,
  signInWithEmailAndPassword,
  signOut,
  onAuthStateChanged,
  updateProfile,
  GoogleAuthProvider,
  signInWithPopup,
  sendPasswordResetEmail,
  type User,
  type Unsubscribe,
} from "firebase/auth";
import { doc, setDoc, getDoc, serverTimestamp } from "firebase/firestore";
import { auth, db, isFirebaseConfigured } from "./firebase";

/** True when Firebase env vars are configured */
export const firebaseEnabled: boolean = isFirebaseConfigured;

// ─── Types ────────────────────────────────────────────────────────────────────

export interface AuthUser {
  uid: string;
  email: string | null;
  displayName: string | null;
  photoURL: string | null;
}

export interface RegisterPayload {
  email: string;
  password: string;
  displayName?: string;
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

function toAuthUser(user: User): AuthUser {
  return {
    uid: user.uid,
    email: user.email,
    displayName: user.displayName,
    photoURL: user.photoURL,
  };
}

/** Save / merge a user profile document in Firestore */
async function upsertProfile(user: User, extra?: Record<string, unknown>) {
  if (!db) return;
  const ref = doc(db, "users", user.uid);
  await setDoc(
    ref,
    {
      uid: user.uid,
      email: user.email,
      displayName: user.displayName,
      photoURL: user.photoURL,
      updatedAt: serverTimestamp(),
      ...extra,
    },
    { merge: true }
  );
}

// ─── Auth operations ─────────────────────────────────────────────────────────

/**
 * Register with email + password.
 * Also creates a Firestore `users/{uid}` document.
 */
export async function registerWithEmail({
  email,
  password,
  displayName,
}: RegisterPayload): Promise<AuthUser> {
  if (!auth) throw new Error("Firebase is not configured. Add VITE_FIREBASE_* variables to Frontend/.env");

  const cred = await createUserWithEmailAndPassword(auth, email, password);

  if (displayName) {
    await updateProfile(cred.user, { displayName });
  }

  await upsertProfile(cred.user, { createdAt: serverTimestamp() });
  return toAuthUser(cred.user);
}

/**
 * Sign in with email + password.
 */
export async function loginWithEmail({
  email,
  password,
}: {
  email: string;
  password: string;
}): Promise<AuthUser> {
  if (!auth) throw new Error("Firebase is not configured. Add VITE_FIREBASE_* variables to Frontend/.env");

  const cred = await signInWithEmailAndPassword(auth, email, password);
  await upsertProfile(cred.user);
  return toAuthUser(cred.user);
}

/**
 * Sign in with Google popup.
 */
export async function loginWithGoogle(): Promise<AuthUser> {
  if (!auth) throw new Error("Firebase is not configured. Add VITE_FIREBASE_* variables to Frontend/.env");

  const provider = new GoogleAuthProvider();
  const cred = await signInWithPopup(auth, provider);
  await upsertProfile(cred.user, { createdAt: serverTimestamp() });
  return toAuthUser(cred.user);
}

/**
 * Sign out the current user.
 */
export async function logoutUser(): Promise<void> {
  if (!auth) return;
  await signOut(auth);
}

/**
 * Send password reset email.
 */
export async function resetPassword(email: string): Promise<void> {
  if (!auth) throw new Error("Firebase is not configured.");
  await sendPasswordResetEmail(auth, email);
}

/**
 * Get the currently signed-in user (sync snapshot).
 * Returns null if not signed in or Firebase is not configured.
 */
export function getCurrentUser(): AuthUser | null {
  if (!auth || !auth.currentUser) return null;
  return toAuthUser(auth.currentUser);
}

/**
 * Get the Firebase ID token for the current user.
 * Used to authenticate requests to the backend.
 * Also aliased as `getAuthToken` for api.js compatibility.
 */
export async function getIdToken(): Promise<string | null> {
  if (!auth || !auth.currentUser) return null;
  return auth.currentUser.getIdToken();
}

/** Alias used by api.js */
export const getAuthToken = getIdToken;

/**
 * Subscribe to auth state changes.
 * Returns an unsubscribe function.
 */
export function watchAuthState(
  callback: (user: AuthUser | null) => void
): Unsubscribe {
  if (!auth) {
    callback(null);
    return () => {};
  }
  return onAuthStateChanged(auth, (user) => {
    callback(user ? toAuthUser(user) : null);
  });
}

/**
 * Fetch the full Firestore user profile for the current user.
 */
export async function getUserProfile(uid: string): Promise<Record<string, unknown> | null> {
  if (!db) return null;
  const snap = await getDoc(doc(db, "users", uid));
  if (!snap.exists()) return null;
  return snap.data() as Record<string, unknown>;
}
