import { createClient } from "@supabase/supabase-js";
import { SUPABASE_ANON_KEY, SUPABASE_BUCKET, SUPABASE_URL } from "./supabaseConfig";

const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

// Uploads the strip image to Supabase Storage and returns a plain public URL.
// The viewer page fetches this URL and triggers the download itself, so we no
// longer force the `download` attachment disposition here — that lets the same
// URL be shown as an on-screen preview before the file is saved.
export async function uploadStripToSupabase(blob, filename) {
  const { error } = await supabase.storage.from(SUPABASE_BUCKET).upload(filename, blob, {
    contentType: "image/png",
    cacheControl: "3600",
    upsert: true,
  });

  if (error) {
    throw error;
  }

  const { data } = supabase.storage.from(SUPABASE_BUCKET).getPublicUrl(filename);
  return data.publicUrl;
}
