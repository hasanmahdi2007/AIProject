import type { Metadata } from "next";
import type { ReactNode } from "react";
import Link from "next/link";
import { JetBrains_Mono, Noto_Sans } from "next/font/google";
import { Disclaimer } from "@/components/Disclaimer";
import { NavLinks } from "@/components/NavLinks";
import "./globals.css";

const sans = Noto_Sans({
  subsets: ["latin"],
  weight: ["400", "600", "700", "800"],
  variable: "--font-sans",
  display: "swap",
});
const mono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Beacon console",
  description:
    "Thin demo console for the Beacon clinical-trial matching engine. Synthetic data only; decision support, not medical advice.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: ReactNode }>) {
  return (
    <html lang="en" className={`${sans.variable} ${mono.variable}`}>
      <body className="min-h-screen font-sans">
        <header className="bg-nhs-dark">
          <div className="mx-auto flex max-w-7xl flex-wrap items-end gap-x-6 gap-y-2 px-4 pt-3 sm:px-6">
            <Link href="/" className="mb-3 flex items-center gap-3">
              <span className="rounded-sm bg-white px-2.5 py-1 text-[22px] font-extrabold leading-7 tracking-tight text-nhs-dark">
                Beacon
              </span>
              <span className="text-[15px] leading-5 text-white">
                clinical-trial matching, with its reasoning shown
              </span>
            </Link>
            <NavLinks />
          </div>
        </header>
        <Disclaimer />
        <main className="mx-auto max-w-7xl px-4 pb-16 pt-8 sm:px-6">{children}</main>
      </body>
    </html>
  );
}
