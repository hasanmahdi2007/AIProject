import type { PatientProfile } from "@/lib/types";

const SEX_TEXT: Record<string, string> = {
  male: "male",
  female: "female",
  all: "unrestricted",
};

function List({ label, values }: { label: string; values: string[] }) {
  return (
    <div>
      <dt className="text-xs font-medium uppercase tracking-wide text-slate-500">
        {label}
      </dt>
      <dd className="mt-0.5 text-sm text-slate-800">
        {values.length > 0 ? (
          <ul className="flex flex-wrap gap-1">
            {values.map((value) => (
              <li
                key={value}
                className="rounded bg-slate-100 px-1.5 py-0.5 text-xs text-slate-800"
              >
                {value}
              </li>
            ))}
          </ul>
        ) : (
          <span className="text-slate-400">not stated in the note</span>
        )}
      </dd>
    </div>
  );
}

/**
 * The structured profile the pipeline extracted, in plain language.
 *
 * Every field is printed as returned. A `null` is shown as "not stated in the
 * note", never filled in or inferred: in this system an unknown is a real
 * value, and it is what the missing-facts lists are built out of.
 */
export function ProfileCard({ profile }: { profile: PatientProfile | undefined }) {
  if (!profile) {
    return (
      <section className="rounded-lg border border-slate-200 bg-white p-4">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-500">
          Patient summary
        </h2>
        <p className="mt-2 text-sm text-slate-600">
          This run did not return a structured profile.
        </p>
      </section>
    );
  }

  const ecog =
    profile.performance_status === null
      ? null
      : `ECOG ${profile.performance_status}`;

  return (
    <section className="rounded-lg border border-slate-200 bg-white p-4">
      <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-500">
        Patient summary
      </h2>
      <p className="mt-2 text-sm leading-6 text-slate-800">
        {profile.age === null ? "Age not stated" : `${profile.age} years old`}
        {profile.sex ? `, ${SEX_TEXT[profile.sex] ?? profile.sex}` : ""}
        {ecog ? `, ${ecog}` : ""}
        {profile.location ? `, in ${profile.location}` : ""}.
      </p>
      <dl className="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-2">
        <List label="Conditions" values={profile.conditions} />
        <List label="Biomarkers" values={profile.biomarkers} />
        <List label="Prior treatments" values={profile.prior_treatments} />
        <List
          label="Labs"
          values={Object.entries(profile.labs).map(
            ([name, value]) => `${name}: ${value}`,
          )}
        />
      </dl>
    </section>
  );
}
