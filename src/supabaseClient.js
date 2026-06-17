import { createClient } from "@supabase/supabase-js";
import { SUPABASE_ANON_KEY, SUPABASE_BUCKET, SUPABASE_URL } from "./supabaseConfig";

const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

// Uploads the strip image to Supabase Storage and returns a public download URL.
// The `download` option makes the URL serve the file as an attachment, so
// scanning the QR on a phone downloads the strip instead of just viewing it.
export async function uploadStripToSupabase(blob, filename) {
  const { error } = await supabase.storage.from(SUPABASE_BUCKET).upload(filename, blob, {
    contentType: "image/png",
    cacheControl: "3600",
    upsert: true,
  });

  if (error) {
    throw error;
  }

  const { data } = supabase.storage.from(SUPABASE_BUCKET).getPublicUrl(filename, { download: true });
  return data.publicUrl;
}
