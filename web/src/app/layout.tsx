import type { Metadata } from "next";
import type { ReactNode } from "react";
import Link from "next/link";
import { Disclaimer } from "@/components/Disclaimer";
import "./globals.css";

export const metadata: Metadata = {
  title: "Beacon console",
  description:
    "Thin demo console for the Beacon clinical-trial matching engine. Synthetic data only; decision support, not medical advice.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: ReactNode }>) {
  return (
    <html lang="en">
      <body className="min-h-screen">
        <Disclaimer />
        <header className="border-b border-slate-200 bg-white">
          <div className="mx-auto flex max-w-6xl items-baseline gap-6 px-4 py-3">
            <Link href="/" className="text-lg font-semibold tracking-tight">
              Beacon
            </Link>
            <span className="text-sm text-slate-500">
              clinical-trial matching, with its reasoning shown
            </span>
            <nav className="ml-auto flex gap-4 text-sm">
              <Link href="/" className="text-slate-600 hover:text-slate-900">
                Console
              </Link>
              <Link
                href="/about"
                className="text-slate-600 hover:text-slate-900"
              >
                How to read this
              </Link>
            </nav>
          </div>
        </header>
        <main className="mx-auto max-w-6xl px-4 py-6">{children}</main>
      </body>
    </html>
  );
}
