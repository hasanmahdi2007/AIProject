import type { PatientProfile } from "@/lib/types";

const SEX_TEXT: Record<string, string> = {
  male: "male",
  female: "female",
  all: "unrestricted",
};

function List({
  label,
  values,
  tone,
}: {
  label: string;
  values: string[];
  tone: string;
}) {
  return (
    <div>
      <dt className="text-[14px] font-bold text-nhs-mid">{label}</dt>
      <dd className="mt-1.5">
        {values.length > 0 ? (
          <ul className="flex flex-wrap gap-1.5">
            {values.map((value) => (
              <li key={value} className={`rounded px-2 py-0.5 text-[14px] font-semibold ${tone}`}>
                {value}
              </li>
            ))}
          </ul>
        ) : (
          <span className="text-[14px] italic text-nhs-mid">not stated in the note</span>
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
export function ProfileCard({ profile }: { profile: PatientProfile | null }) {
  if (!profile) {
    return (
      <section className="card p-5">
        <h2 className="heading">Patient summary</h2>
        <p className="mt-2 text-[14px] leading-6 text-nhs-mid">
          No structured profile yet — the run has not finished profiling the
          note. Nothing is wrong with the run.
        </p>
      </section>
    );
  }

  const ecog =
    profile.performance_status === null
      ? null
      : `ECOG ${profile.performance_status}`;

  return (
    <section className="card p-5 sm:p-6">
      <h2 className="heading">Patient summary</h2>
      <p className="mt-1 text-[22px] font-bold leading-8 text-nhs-ink">
        {profile.age === null ? "Age not stated" : `${profile.age} years old`}
        {profile.sex ? `, ${SEX_TEXT[profile.sex] ?? profile.sex}` : ""}
        {ecog ? `, ${ecog}` : ""}
        {profile.location ? `, in ${profile.location}` : ""}.
      </p>
      <dl className="mt-4 grid grid-cols-1 gap-x-6 gap-y-4 border-t border-nhs-line pt-4 sm:grid-cols-2">
        <List label="Conditions" values={profile.conditions} tone="bg-[#DCE9F6] text-nhs-dark" />
        <List label="Biomarkers" values={profile.biomarkers} tone="bg-nhs-aquapale text-[#00445E]" />
        <List label="Prior treatments" values={profile.prior_treatments} tone="bg-[#E7E0F0] text-nhs-purple" />
        <List
          label="Labs"
          values={Object.entries(profile.labs).map(([name, value]) => `${name}: ${value}`)}
          tone="bg-nhs-pale font-mono text-nhs-ink"
        />
      </dl>
    </section>
  );
}
